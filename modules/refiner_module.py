import lightning as L
import torch
from einops import rearrange
from modules.models.FNO1Dcond import FNO1d_bundled_cond
from modules.models.Unet1Dcond import Unet1D_cond
from modules.models.FNO2Dcond import FNO2d_cond
from modules.models.Unet2Dcond import Unet2D_cond
from diffusers.schedulers import DDPMScheduler
from common.loss import ScaledLpLoss, PearsonCorrelationScore

class RefinerModule(L.LightningModule):
    def __init__(self,
                 modelconfig: dict,):
        '''
        RefinerModule to implement PDE Refiner
        https://github.com/pdearena/pdearena/blob/main/pdearena/models/pderefiner.py
        args:
            modelconfig: dict, model configuration
        '''

        super().__init__()
        self.model_name = modelconfig["model_name"]
        self.min_noise_std = modelconfig["min_noise_std"]
        self.num_refinement_steps = modelconfig["num_refinement_steps"]
        self.lr = modelconfig["lr"]
        self.train_mode = modelconfig["train_mode"]
        self.inference_mode = modelconfig["inference_mode"]
        self.correlation = modelconfig["correlation"]

        self.mse_criterion = torch.nn.MSELoss()
        self.criterion = ScaledLpLoss()
        self.correlation_criterion = PearsonCorrelationScore(reduce_batch=True)

        # We use the Diffusion implementation here. Alternatively, one could
        # implement the denoising manually.
        betas = [self.min_noise_std ** (k / self.num_refinement_steps) for k in reversed(range(self.num_refinement_steps + 1))]
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )
        # Multiplies k before passing to frequency embedding.
        self.time_multiplier = 1000 / self.num_refinement_steps

        if self.model_name == "fno":
            fnoconfig = modelconfig["fno"]
            fnoconfig['in_channels'] = fnoconfig['in_channels'] * 2 # double input channels for noise
            self.model = FNO1d_bundled_cond(**fnoconfig)
        elif self.model_name == "unet":
            unetconfig = modelconfig["unet"]
            unetconfig['n_input_scalar_components'] = unetconfig['n_input_scalar_components'] * 2 # double input channels for noise
            self.model = Unet1D_cond(**unetconfig)
        elif self.model_name == "fno2d":
            fnoconfig = modelconfig["fno2d"]
            fnoconfig['in_channels'] = fnoconfig['in_channels'] * 2 
            self.model = FNO2d_cond(**fnoconfig)
        elif self.model_name == "unet2d":
            unetconfig = modelconfig["unet2d"]
            unetconfig['n_input_scalar_components'] = unetconfig['n_input_scalar_components'] * 2
            self.model = Unet2D_cond(**unetconfig)
        else:
            raise ValueError("Model not found")
        
        self.model.time_scale = self.time_multiplier # set time multiplier in model

        print(f"Training: {self.model_name} with PDE Refiner")
    

    def forward(self, x, cond):
        return self.predict_next_solution(x, cond)

    def get_data_labels(self, u, idx):
        '''
        Get data and labels for training
        args:
            u: shape (b, nt, nx) or (b, nt, nx, ny) or (b, nt, nx, ny, c)
            idx: shape (b,)
        returns:
            data: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c)
            label: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c)
        '''
        b = u.shape[0]
        batch_range = torch.arange(b)

        data = u[batch_range, idx] # get u(t), shape (b, nx)
        label = u[batch_range, idx+1] # gets u(t+1), shape (b, nx)

        if len(data.shape) < 4: # add channel dimension
            data = data.unsqueeze(-1) # shape (b, nx, 1) or (b, nx, ny, 1)
            label = label.unsqueeze(-1) # shape (b, nx, 1) or (b, nx, ny, 1)

        return data, label
    
    def train_step(self, batch):
        u = batch['u']
        b, nt, = u.shape[0], u.shape[1] # u is in shape (b t nx) or (b t nx ny)
        cond = batch['cond'] # shape (b, 1)

        rand_idx = torch.randint(0, nt-1, (b,)) # shape (b,) get random start indexes. idx < nt-1

        # x is u(t), y is u(t+1)
        x, y = self.get_data_labels(u, rand_idx) # shape (b, nx, 1) or (b, nx, ny, 1)
        
        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device)
        noise_factor = self.scheduler.alphas_cumprod.to(x.device)[k]
        noise_factor = noise_factor.view(-1, *[1 for _ in range(x.ndim - 1)])
        signal_factor = 1 - noise_factor
        noise = torch.randn_like(y) # shape (b, nx, 1) or (b, nx, ny, 1)
        y_noised = self.scheduler.add_noise(y, noise, k)

        x_in = torch.cat([x, y_noised], axis=-1) # shape (b, nx, 2) or (b, nx, ny, 2)
        pred = self.model(x_in, t=k, c=cond) # time will be scaled in the model fwd call, shape (b, nx, 1) or (b, nx, ny, 1)

        target = (noise_factor**0.5) * noise - (signal_factor**0.5) * y
        loss = self.mse_criterion(pred, target) # take MSE for diffusion training
        return loss

    def training_step(self, batch, batch_idx: int):
        loss = self.train_step(batch)
        self.log("train_mse_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_next_solution(self, x, cond):
        if len(x.shape) == 3:
            b, nx, c = x.shape # x in shape b nx 1
            y_noised = torch.randn(
                size=(b, nx, c), dtype=x.dtype, device=x.device
            )
        else:
            b, nx, ny, c = x.shape
            y_noised = torch.randn(
                size=(b, nx, ny, c), dtype=x.dtype, device=x.device
            )

        for k in self.scheduler.timesteps:
            time = torch.zeros(size=(x.shape[0],), dtype=x.dtype, device=x.device) + k
            x_in = torch.cat([x, y_noised], axis=-1) # shape (b, nx, 2)
            pred = self.model(x_in, t=time, c=cond)
            y_noised = self.scheduler.step(pred, k, y_noised).prev_sample
        y = y_noised
        return y
    
    def compute_rolloutloss(self, batch):
        # Can be simplified since only using time_history and time_future = 1 

        u = batch['u']
        b, nt, = u.shape[0], u.shape[1] # u is in shape (b t nx) or (b t nx ny)
        cond = batch['cond'] # shape (b, 1)

        u_input = u[:, 0].unsqueeze(-1) # shape (b, nx, 1) or (b, nx, ny, 1)
        u_pred = torch.zeros_like(u) # shape (b, nt, nx) or (b, nt, nx, ny)
        u_pred[:, 0] = u_input.squeeze() # set initial condition

        accumulated_loss = []
        at_correlation = False

        for i in range(0, nt-1):
            u_input = self.forward(u_input, cond) # (b, nx, 1) or (b, nx, ny, 1)
            u_true = u[:, i+1].unsqueeze(-1) # shape (b, nx, 1) or (b, nx, ny, 1)
            loss = self.criterion(u_input, u_true) # calculate loss
            accumulated_loss.append(loss.item())

            correlation = self.correlation_criterion(u_input, u_true) # calculate correlation
            if correlation < self.correlation and not at_correlation:
                correlation_time = float(i) # get time step at correlation
                at_correlation = True 

            u_pred[:, i+1] = u_input.squeeze()

        if not at_correlation:
            correlation_time = nt-1 # didn't go below correlation threshold, therefore the time is the last step
            
        return accumulated_loss, correlation_time, u, u_pred

    def validation_step(self, batch, batch_idx, eval=False):
        accumulated_loss, correlation_time, u, u_pred = self.compute_rolloutloss(batch)
        
        if not eval:
            avg_loss = sum(accumulated_loss) / len(accumulated_loss)
            self.log("rollout_loss", avg_loss, on_step=False, on_epoch=True)
            self.log("correlation_time", correlation_time, on_step=False, on_epoch=True)
        else:
            return accumulated_loss, correlation_time, u, u_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        return [optimizer]