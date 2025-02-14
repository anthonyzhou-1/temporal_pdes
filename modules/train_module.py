import lightning as L
import torch
from modules.models.FNO1Dcond import FNO1d_bundled_cond
from modules.models.Unet1Dcond import Unet1D_cond
from modules.models.FNO2Dcond import FNO2d_cond 
from modules.models.Unet2Dcond import Unet2D_cond
from common.utils import forward_difference, central_difference, richardson_extrapolation
from common.loss import ScaledLpLoss, PearsonCorrelationScore
import copy

class TrainModule(L.LightningModule):
    def __init__(self,
                 modelconfig: dict,):
        '''
        TrainModule for implementing different training and inference strategies
        During training, either seeks to match the solution at the next timestep (u(t+1)), or the derivative at the current timestep (u'(t)). 
        The derivative can be numerically approximated with different schemes:
            - forward difference (O(dt))
            - central difference (O(dt^2))
            - Richardson extrapolation (O(dt^4))
        During inference, the model can step forward in time using different schemes:
            - normal: directly predict u(t+1)
            - forward_euler: step forward using u(t) + dt * u'(t)  
            - adams_bashforth: step forward using u(t) + dt * (3/2 * u'(t) - 1/2 * u'(t-1))
            - heun: step forward using u(t) + dt/2 * (u'(t) + u'(t+1))
            - rk4: step forward using u(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4), where k1, k2, k3, k4 are RK4 coefficients
        Additionally implements pushforward trick for comparison
        args:
            modelconfig: dict, model configuration
        '''

        super().__init__()
        self.model_name = modelconfig["model_name"]
        self.lr = modelconfig["lr"]
        self.train_mode = modelconfig["train_mode"]
        self.inference_mode = modelconfig["inference_mode"]
        self.correlation = modelconfig["correlation"]
        self.k = modelconfig["k"] if "k" in modelconfig else 1 # control intermediate stepping with k
        self.dt_jump = modelconfig["dt_jump"] if "dt_jump" in modelconfig else 1 # control intermediate stepping with dt_jump

        self.criterion = ScaledLpLoss()
        self.correlation_criterion = PearsonCorrelationScore(reduce_batch=True)

        self.pushforward_steps = modelconfig["pushforward_steps"] if "pushforward_steps" in modelconfig else 0
        self.pushforward_grad = modelconfig["pushforward_grad"] if "pushforward_grad" in modelconfig else False
        # Let model train w/o pf for a few epochs. PF can be very noisy in the beginning
        self.warmup_epochs = modelconfig["warmup_epochs"] if "warmup_epochs" in modelconfig else 999

        if self.model_name == "fno":
            fnoconfig = modelconfig["fno"]
            self.model = FNO1d_bundled_cond(**fnoconfig)
        elif self.model_name == "unet":
            unetconfig = modelconfig["unet"]
            self.model = Unet1D_cond(**unetconfig)
        elif self.model_name == "fno2d":
            fnoconfig = modelconfig["fno2d"]
            self.model = FNO2d_cond(**fnoconfig)
        elif self.model_name == "unet2d":
            unetconfig = modelconfig["unet2d"]
            self.model = Unet2D_cond(**unetconfig)
        else:
            raise ValueError("Model not found")

        print(f"Training: {self.model_name}, with train_mode: {self.train_mode}, and inference_mode: {self.inference_mode}")
        print(f"Pushforward steps: {self.pushforward_steps}, k: {self.k}, dt_jump: {self.dt_jump}, warmup_epochs: {self.warmup_epochs}, pushforward_grad: {self.pushforward_grad}")

    def forward(self, u, t=None, cond=None):
        return self.model(u, t, cond)

    def get_data_labels(self, u, idx, dt, mode='normal'):
        '''
        Get data and labels for training
        Can use different schemes for different accuracies of approximating u'(t)
        args:
            u: shape (b, nt, nx) or (b, nt, nx, ny) or (b, nt, nx, ny, c)
            idx: shape (b,)
            dt: float (assumed constant)
            mode: str, training mode
        returns:
            data: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c)
            label: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c)
        '''
        b = u.shape[0]

        batch_range = torch.arange(b)

        data = u[batch_range, idx] # get u(t), shape (b, nx) or (b, nx, ny)

        if mode == 'normal' or mode == "pushforward":
            label = u[batch_range, idx+1] # gets u(t+1), shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
        elif mode == "forward_difference":
            label = forward_difference(u, idx, dt) # gets u'(t), shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
        elif mode == "central_difference":
            label = central_difference(u, idx, dt) # gets u'(t), shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
        elif mode == "richardson_extrapolation":
            label = richardson_extrapolation(u, idx, dt) # gets u'(t), shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
        else:
            raise ValueError("Mode not found")

        if len(data.shape) < 4: # add channel dimension
            data = data.unsqueeze(-1) # shape (b, nx, 1) or (b, nx, ny, 1)
            label = label.unsqueeze(-1) # shape (b, nx, 1) or (b, nx, ny, 1)

        return data, label
            
    def get_time(self, batch, idx):
        t = batch['t'] # shape (b, nt)
        batch_range = torch.arange(t.shape[0])
        t_idx = t[batch_range, idx] # shape (b,) for each sample in the batch, get the time at idx 
        return t_idx
    
    def inference_step(self, u_start, pred, dt, mode="normal", pred_cache=None, tplus1=None, tplus2=None, cond=None):
        '''
        Steps u(t) to u(t+1), using u_start and model prediction
        args:
            u_start: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c), u(t)
            pred: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c), model prediction F(u(t), cond) = u'(t)
            dt: float, step size 
            mode: str, inference mode
            pred_cache: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c), cached previous prediction u'(t-1), used in multistep methods
            tplus1: shape (b,), time at t+1,
            tplus2: shape (b,), time at t+2,
            cond: shape (b,), condition at t, doesn't change with time
        returns:
            u_pred: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c), prediction at t+1
        '''

        if mode == "normal" or mode == "pushforward":
            return pred # pred is already u(t+1)
        
        elif mode == "forward_euler":
            return u_start + dt * pred # u(t+1) = u(t) + dt * u'(t)
        
        elif mode == "adams_bashforth":
            if pred_cache is None: # need u'(t-1) for adams_bashforth
                return u_start + dt * pred # default to forward euler
            else:
                return u_start + dt * (3/2 * pred - 1/2 * pred_cache) # u(t+1) = u(t) + dt * (3/2 * u'(t) - 1/2 * u'(t-1))
        
        elif mode == "heun":
            u_intermediate = u_start + dt * pred # u(t+1) = u(t) + dt * u'(t)
            pred_intermediate = self.model(u_intermediate, tplus1, cond) # u'(t+1)
            return u_start + dt/2 * (pred + pred_intermediate) # u(t+1) = u(t) + dt/2 * (u'(t) + u'(t+1))
        
        elif mode == "rk4": 
            # Assumes that tplus1 is actually t+dt/2, and tplus2 is t+dt          
            k1 = pred # u'(t)
            k2 = self.model(u_start + dt/2 * k1, tplus1, cond) # F(u(t) + dt/2 * k1, t+dt/2)
            k3 = self.model(u_start + dt/2 * k2, tplus1, cond) # F(u(t) + dt/2 * k2, t+dt/2)
            k4 = self.model(u_start + dt * k3, tplus2, cond) # F(u(t) + dt * k3, t+dt)
            return u_start + dt/6 * (k1 + 2*k2 + 2*k3 + k4) # u(t+1) = u(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError("Mode not found")
        
    def rollout(self, batch, k=1):
        # control intermediate stepping with k 
        u = batch['u']
        dt = batch['dt'][0] # assume constant dt across samples in batch
        dt = dt / k 

        if self.dt_jump > 1:
            dt = dt * self.dt_jump  
            u = u[:, ::self.dt_jump] # downsample u by dt_jump

        b = u.shape[0]
        nt = u.shape[1]
        cond = batch['cond'] # shape (b,) or (b, n_cond) depending on dimensionality of condition
        u_input = u[:, 0]

        if len(u_input.shape) < 4: # add channel dimension
            u_input = u_input.unsqueeze(-1) # shape (b, nx, 1) or (b, nx, ny, 1), starting at t_idx = 0

        u_pred = torch.zeros_like(u) # shape (b, nt, nx) or (b, nt, nx, ny) or (b, nt, nx, ny, c)
        u_pred[:, 0] = u[:, 0] # set initial condition

        accumulated_loss = []
        at_correlation = False

        pred_cache = None 
        tplus1 = None
        tplus2 = None

        last_step = k*(nt-1)# rollout to nt-1, since t = 0 is given. Step by k internally
        last_step = last_step
        t = self.get_time(batch, torch.zeros(b, dtype=torch.long)) # shape (b,) get physical time at t_idx = 0

        for i in range(0, last_step):
            pred = self.model(u_input, t, cond) # shape (b, nx, 1) or (b, nx, ny, 1)

            if self.inference_mode == "heun":
                tplus1 = t + dt # get time at t+1
            elif self.inference_mode == "rk4":
                tplus2 = t + dt
                tplus1 = t + dt/2

            u_input = self.inference_step(u_input,
                                            pred,
                                            dt,
                                            mode=self.inference_mode,
                                            pred_cache=pred_cache,
                                            tplus1=tplus1,
                                            tplus2=tplus2,
                                            cond=cond) # shape (b, nx, 1)

            if (i+1) % k == 0: # only save/calculate loss every k steps
                save_idx = (i+1) // k
                u_pred[:, save_idx] = u_input.squeeze() # save prediction
                u_true = u[:, save_idx]
                if len(u_true.shape) < 4: # add channel dimension
                    u_true = u_true.unsqueeze(-1)
                loss = self.criterion(u_input, u_true) # calculate loss
                accumulated_loss.append(loss.item())

                correlation = self.correlation_criterion(u_input, u_true) # calculate correlation
                if correlation < self.correlation and not at_correlation:
                    correlation_time = float(save_idx) # get time step at correlation
                    at_correlation = True 

            t = t + dt # increment time
            pred_cache = pred # cache prediction for adams_bashforth
        
        if not at_correlation:
            correlation_time = nt-1 # didn't go below correlation threshold, therefore the time is the last step

        return accumulated_loss, correlation_time, u, u_pred
    
    def training_step(self, batch, batch_idx):
        u = batch['u']
        dt = batch['dt'][0] # assume constant dt across samples in batch
        b = u.shape[0]
        nt = u.shape[1]

        t_idx = torch.randint(0, nt-1-self.pushforward_steps, (b,)) # shape (b,) get random start indexes. idx < nt-1-pushforward_steps
        t = self.get_time(batch, t_idx) # shape (b,) get physical time at t_idx
        cond = batch['cond'] # shape (b,) or (b, n_cond) depending on dimensionality of condition

        data, labels = self.get_data_labels(u, t_idx, dt, mode=self.train_mode) # slice data and labels with t_idx
        pred_cache = None
        tplus1 = None
        tplus2 = None

        if self.current_epoch > self.warmup_epochs:
            with torch.set_grad_enabled(self.pushforward_grad): # if pushforward_grad is false, don't calculate grads through intermediate steps
                for i in range(self.pushforward_steps):
                    t_idx = t_idx + 1 # increment time idx
                    _, labels = self.get_data_labels(u, t_idx, dt, mode=self.train_mode) # label is at t_idx+2

                    # data, t are all at t_idx
                    pred = self.model(data, t, cond) 

                    if self.inference_mode == "heun":
                        tplus1 = t + dt # get time at t+1
                    elif self.inference_mode == "rk4":
                        tplus2 = t + dt
                        tplus1 = t + dt/2

                    # data, t are incremented to t_idx+1
                    data = self.inference_step(data,
                                            pred,
                                            dt,
                                            mode=self.inference_mode,
                                            pred_cache=pred_cache,
                                            tplus1=tplus1,
                                            tplus2=tplus2,
                                            cond=cond) # shape (b, nx, 1)
                    t = self.get_time(batch, t_idx) # get time at t_idx + 1

                    pred_cache = pred # cache prediction for adams_bashforth

        target = self.model(data, t, cond) # shape (b, nx, 1) or (b, nx, ny, 1)
            
        loss = self.criterion(target, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, eval=False):
        if eval: 
            accumulated_loss, correlation_time, u, u_pred = self.rollout(batch, self.k)
            return accumulated_loss, correlation_time, u, u_pred

        else:
            accumulated_loss, correlation_time, _, _ = self.rollout(batch)
            avg_loss = sum(accumulated_loss) / len(accumulated_loss)
            self.log("rollout_loss", avg_loss, on_step=False, on_epoch=True)
            self.log("correlation_time", correlation_time, on_step=False, on_epoch=True)

            k = copy.deepcopy(self.k)
            while(k // 2 >= 1):
                accumulated_loss, correlation_time, _, _ = self.rollout(batch, k)
                avg_loss = sum(accumulated_loss) / len(accumulated_loss)
                self.log(f"rollout_loss_{k}", avg_loss, on_step=False, on_epoch=True)
                self.log(f"correlation_time_{k}", correlation_time, on_step=False, on_epoch=True)
                k = k // 2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

        return [optimizer], [scheduler]
    