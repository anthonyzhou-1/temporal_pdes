import torch 

import abc
from einops import rearrange
from loss_landscapes.model_interface.model_wrapper import ModelWrapper

class Metric(abc.ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass

class PDELoss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, pde: str, mode: str):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.pde=pde
        self.mode=mode

    def get_data_labels(self, u, idx, mode='normal'):
        b, nt, nx = u.shape

        data = u[torch.arange(b), idx] # get the start time step, shape (b, nx)

        if mode == 'normal':
            label = u[torch.arange(b), idx+1] # get the next time step, shape (b, nx)
        elif mode == "residual":
            label = u[torch.arange(b), idx+1] - data # get the residual, shape (b, nx)
        else:
            raise ValueError("Mode not found")

        data = rearrange(data, 'b nx -> b nx 1') # shape (b, nx, 1)
        label = rearrange(label, 'b nx -> b nx 1')

        return data, label
    
    def get_cond(self, variables):
        if self.pde == "advection":
            # advection speed (b, 1)
            cond = variables['a'].unsqueeze(1)  
        elif self.pde == "heat":
            # diffusion coefficient (b, 1)
            cond = variables['beta'].unsqueeze(1)
        elif self.pde == "ks":
            # viscosity (b, 1)
            cond = variables['v'].unsqueeze(1)
        else:
            raise ValueError("PDE not found")
        
        return cond

    def __call__(self, model_wrapper: ModelWrapper) -> float:

        # assume input is in shape u, x, variables
        u, _, variables = self.inputs
        b, nt, nx = u.shape
        cond = self.get_cond(variables)
        accumulated_loss = []
        idx = torch.zeros(u.shape[0], dtype=torch.int64)
        data, _ = self.get_data_labels(u, idx) # data at t0 

        for i in range(0, nt-1):
            x = (data, cond)
            target = model_wrapper.forward(x) # target at t_i+1
            _, labels = self.get_data_labels(u, idx) # get label at t_i+1

            if self.mode == 'residual':
                target = target + data # if predicting residual, add to previous step
            
            loss = self.loss_fn(target, labels) # calculate loss
            accumulated_loss.append(loss.item())

            data = target # data at t_i+1
            idx = idx + 1 # increment index

        avg_loss = sum(accumulated_loss) / nt

        return avg_loss

class PDEModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__([model])

    def forward(self, x):
        data, cond = x
        return self.modules[0](data, cond)