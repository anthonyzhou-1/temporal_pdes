import h5py
import torch
from torch.utils.data import Dataset
import numpy as np 
from typing import Tuple

# Using different dataset objects due to different raw data formats and processing steps

class PDEDataset2D(Dataset):
    """Load samples of a 2D PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 split: str,
                 resolution: list=None,
                 norm_vars = True,
                 load_mem = False,
                 start=0.0) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            split: [train, valid, test]
            resolution: resolution of the dataset [nt, nx]
            norm_vars: normalize variables
            load_mem: load data to memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.split = split
        self.pde = pde
        self.resolution = resolution
        self.nt, self.nx, self.ny = resolution
        data = f[self.split]
        self.norm_vars = norm_vars
        self.n_samples = len(data['u'])
        self.load_mem = load_mem
        self.start = start 

        self.variables, self.var_range = self.get_variables(self.pde)
        self.attrs = ['u'] + self.variables
        if load_mem:
            for attr in self.attrs + ['x', 't']:
                setattr(self, attr, 
                        torch.as_tensor(np.array(data[attr]), 
                                        dtype=torch.float32)
                        )
        else:
            for attr in self.attrs + ['x', 't']:
                setattr(self, attr, data[attr])
        
        if load_mem:
            f.close()

        nt_data = self.t.shape[0] # original nt
        self.start_t = int(self.start * nt_data) # start time index
        self.t_downsample = int(nt_data / self.nt)  # downsample factor 

        self.t = self.t[self.start_t:] # start from start
        self.t = self.t[::self.t_downsample] # downsample time 

        #assert len(self.t) == int((1-self.start) * self.nt), "Time dimension mismatch"

        print(f"t is starting from {self.t[0]}, at step {self.start_t} of the original data")
        self.t = self.t - self.t[0] # start time from zero
        self.dt = self.t[1] - self.t[0]
        self.dx = self.x[0, 0, 1] - self.x[0, 0, 0]
        self.dy = self.x[1, 1, 0] - self.x[1, 0, 0]

        self.nt = len(self.t)

        print("Data loaded from: {}".format(path))
        print(f"PDE: {self.pde}, dt: {self.dt:.3f}, dx: {self.dx:.3f}, nt: {self.nt}, nx: {self.nx}")
        print(f"Time ranges from {self.t[0]:.3f} to {self.t[-1]:.3f} = {self.dt:.3f} * {self.nt} dt * nt")
        print("\n")

    def __len__(self):
        return self.n_samples
    
    def get_variables(self, pde):
        if pde == "burgers_2d":
            variables = ['nu', 'cx', 'cy']
            ranges = {'nu': [7.5e-3, 1.5e-2],
                    'cx': [0.5, 1.0],
                    'cy': [0.5, 1.0]}
        elif pde == "ns_2d":
            variables = [] # no cond
            ranges = {}
        else:
            raise ValueError("PDE not found")
        
        return variables, ranges
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, list]:
        """
        Get data item
        Args:
            i (int): data index
        Returns:
            u: torch.Tensor: numerical baseline trajectory of size [nt, nx, ny]
            cond: torch.Tensor: condition in shape [n_vars]
            dx: float: spatial resolution
            dt: float: temporal resolution
            t: torch.Tensor: time in shape [nt]
        """
        
        idx = i
        variables = {attr: getattr(self, attr)[idx] for attr in self.attrs}
        u = variables.pop('u') # nt nx ny

        var_list = []
        for var in self.variables:
            if self.norm_vars and self.var_range is not None:
                var_norm = (variables[var] - self.var_range[var][0]) / (self.var_range[var][1] - self.var_range[var][0])
                if not self.load_mem:
                    var_norm = torch.as_tensor(var_norm, dtype=torch.float32)
                var_list.append(var_norm)
            else:
                var_app = variables[var]
                if not self.load_mem:
                    var_app = torch.as_tensor(var_app, dtype=torch.float32)
                var_list.append(var_app)
        
        if len(var_list) == 0:
            cond = 1 # no condition
        else:
            cond = torch.stack(var_list) # n_vars

        u = u[self.start_t:]
        u = u[::self.t_downsample] # truncate to nt by taking every t_downsample
        return_dict = {"u": u, "cond": cond, "dx": self.dx, "dt": self.dt, "t": self.t}
        return return_dict
