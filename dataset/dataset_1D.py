import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class PDEDataset1D(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 split: str,
                 resolution: Tuple[int, int],
                 norm_dims: bool = False,
                 norm_vars: bool = True) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            split: [train, valid, test]
            resolution: resolution of the dataset [nt, nx]
            norm_dims: normalize x and t
            norm_vars: normalize variables
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.split = split
        self.pde = pde
        self.resolution = resolution
        self.nt, self.nx = resolution
        self.norm_dims = norm_dims
        self.norm_vars = norm_vars
        data = f[self.split]
        self.n_samples = len(data['u'])

        self.variables, self.var_range = self.get_variables(self.pde)
        self.attrs = ['u'] + self.variables
        for attr in self.attrs + ['x', 't']:
            setattr(self, attr, 
                    torch.as_tensor(np.array(data[attr]), 
                                    dtype=torch.float32)
                    )
        f.close()

        # let time and x start from zero for consistent embeddings
        if len(self.t.shape) > 1:
            self.t = self.t[0] # assume constant time across batch, take first one
        if len(self.x.shape) > 1:
            self.x = self.x[0]

        self.t = self.t - self.t[0] # start time from zero
        self.x = self.x - self.x[0] # start x from zero

        nt_data = self.t.shape[0]
        self.t_downsample = int(nt_data / self.nt) 
        self.t = self.t[::self.t_downsample] # downsample time

        # normalize time and x. Optional, but will change dt and dx
        if self.norm_dims:
            self.t = self.t / self.t[-1]
            self.x = self.x / self.x[-1]

        self.dt = self.t[1] - self.t[0]
        self.dx = self.x[1] - self.x[0]

        print("Data loaded from: {}".format(path))
        print(f"PDE: {self.pde}, dx: {self.dx:.3f}, nt: {self.nt}, nx: {self.nx}, downsample: {self.t_downsample}")
        print(f"Time ranges from {self.t[0]:.3f} to {self.t[-1]:.3f} = {self.dt:.3f} * {self.nt} dt * nt")

    def __len__(self):
        return self.n_samples
    
    def get_variables(self, pde):
    
        if pde == "heat":
            variables = ['beta']
            ranges = {'beta': [0.1, 0.8]}
        elif pde == "advection":
            variables = ['a']
            ranges = {'a': [0.1, 2.5]} 
        elif pde == "ks":
            variables = ['v']
            ranges = {'v': [1.0, 1.0]}
        else:
            raise ValueError("PDE not found")

        return variables, ranges  

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            u: torch.Tensor: numerical baseline trajectory of size [nt, nx]
            cond: torch.Tensor: condition in shape [1]
            dx: float: spatial resolution
            dt: float: temporal resolution
        """
        u = self.u[idx]
        var_name = self.variables[0]
        cond = getattr(self, var_name)[idx] # assume only one variable

        if self.norm_vars:
            var_max, var_min = self.var_range[var_name]
            cond = (cond - var_min) / (var_max - var_min) # scale to [0, 1]

        if self.pde == "heat":
            u = torch.flip(u, [0]) # flip along time axis since mistake in data generation
        u = u[::self.t_downsample] # truncate to nt by taking every t_downsample

        return_dict = {"u": u, "cond": cond, "dx": self.dx, "dt": self.dt, "x": self.x, "t": self.t}

        return return_dict