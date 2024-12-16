import yaml
import torch 
import math 

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_yaml(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def sinusoidal_embedding(timesteps, dim, max_period=10000, scale=1):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    timesteps = timesteps * scale # scale the timesteps since in our case they can be very small

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def forward_difference(u, idx, dt):
    '''
    Compute the forward difference of a 1D tensor at timestep idx
    In effect, computing u'(t)|t=idx, but idx is random
    args:
        u: shape (b, nt, nx) or (b, nt, nx, ny) or (b, nt, nx, ny, c)
        idx: shape (b,)
        dt: scalar
    returns:
        du_dt: shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
    '''
    b = u.shape[0]
    nt = u.shape[1]

    batch_range = torch.arange(b)
    last_idx = nt - 1 # nt is length of u, so nt-1 is the last index (i.e. u[:,nt] is out of bounds)

    is_not_n_minus_1 = (idx-(last_idx)).all() # true if all elements are not nt-1, false if at least one element is nt-1 

    if is_not_n_minus_1:
        # can compute batched forward
        u_t = u[batch_range, idx] # get u(t), shape (b, nx)
        # idx is guaranteed to be less than nt-1, so can do this
        u_tplus1 = u[batch_range, idx+1] # get u(t+1), shape (b, nx)
        du_dt = (u_tplus1 - u_t) / dt
    
    else:
        # need to elementwise compute forward difference 
        u_shape_no_time = u[:, 0].shape
        du_dt = torch.zeros(u_shape_no_time, device=u.device)
        for i in range(b):
            idx_i = idx[i]
            if idx_i == last_idx: # compute backward difference
                u_tminus1 = u[i, idx_i-1]
                du_dt[i] = (u[i, idx_i] - u_tminus1) / dt
            else:
                u_tplus1 = u[i, idx_i+1]
                du_dt[i] = (u_tplus1 - u[i, idx_i]) / dt

    return du_dt

def central_difference(u, idx, dt):
    '''
    Compute the central difference of a 1D tensor at timestep idx
    In effect, computing u'(t)|t=idx, but idx is random
    args:
        u: shape (b, nt, nx) or (b, nt, nx, ny) or (b, nt, nx, ny, c)
        idx: shape (b,)
        dt: scalar
    returns:
        du_dt: shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
    '''
    b = u.shape[0]
    nt = u.shape[1]
    batch_range = torch.arange(b)
    last_idx = nt - 1 # nt is length of u, so nt-1 is the last index (i.e. u[:,nt] is out of bounds)

    is_not_n_minus_1 = (idx-(last_idx)).all() # true if all elements are not nt-1, false if at least one element is nt-1 
    is_not_zero = idx.all() # true if all elements are not zero, false if at least one is zero

    if is_not_zero and is_not_n_minus_1:
        # can compute batched central difference
        u_tminus1 = u[batch_range, idx-1] # get u(t-1), shape (b, nx)
        u_tplus1 = u[batch_range, idx+1] # get u(t+1), shape (b, nx)
        du_dt = (u_tplus1 - u_tminus1) / (2*dt)
    else:
        # need to elementwise compute central difference since idx-1 < 0 
        u_shape_no_time = u[:, 0].shape
        du_dt = torch.zeros(u_shape_no_time, device=u.device)
        for i in range(b):
            idx_i = idx[i]
            if idx_i == 0: # compute second order one-sided difference
                u_t = u[i, idx_i]
                u_tplus1 = u[i, idx_i+1]
                u_tplus2 = u[i, idx_i+2]
                du_dt[i] = -1 * (3 * u_t - 4 * u_tplus1 + u_tplus2) / (2*dt)
            elif idx_i == last_idx: # compute second order one-sided difference
                u_t = u[i, idx_i]
                u_tminus1 = u[i, idx_i-1]
                u_tminus2 = u[i, idx_i-2]
                du_dt[i] = (3 * u_t - 4 * u_tminus1 + u_tminus2) / (2*dt)
            else:
                u_tminus1 = u[i, idx_i-1]
                u_tplus1 = u[i, idx_i+1]
                du_dt[i] = (u_tplus1 - u_tminus1) / (2*dt)

    return du_dt

def richardson_extrapolation(u, idx, dt):
    '''
    Compute the Richardson extrapolation of a 1D tensor at timestep idx
    In effect, computing u'(t)|t=idx
        args:
        u: shape (b, nt, nx) or (b, nt, nx, ny) or (b, nt, nx, ny, c)
        idx: shape (b,)
        dt: scalar
    returns:
        du_dt: shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
    '''
    b = u.shape[0]
    nt = u.shape[1]
    batch_range = torch.arange(b)
    last_idx = nt-1 # nt is length of u, so nt-1 is the last index (i.e. u[:,nt] is out of bounds)

    is_not_zero = idx.all() # true if all elements are not zero, false if at least one element is zero
    is_not_one = (idx-1).all() # true if all elements are not one, false if at least one element is one
    is_not_n_minus_1 = (idx-(last_idx)).all() # true if all elements are not nt-1, false if at least one element is nt-1 
    is_not_n_minus_2 = (idx-(last_idx-1)).all() # true if all elements are not nt-2, false if at least one element is nt-2  
    # assume that idx is less than nt-1, so dont need to check if idx+1 is out of bounds 

    if is_not_zero and is_not_one and is_not_n_minus_2 and is_not_n_minus_1:
        # can compute batched Richardson extrapolation
        u_tminus1 = u[batch_range, idx-1]
        u_tminus2 = u[batch_range, idx-2]
        u_tplus1 = u[batch_range, idx+1]
        u_tplus2 = u[batch_range, idx+2]

        du_dt = 4/3 * (u_tplus1 - u_tminus1)/(2*dt) - 1/3 * (u_tplus2 - u_tminus2)/(4*dt)
    else:
        # compute elementwise Richardson extrapolation
        u_shape_no_time = u[:, 0].shape
        du_dt = torch.zeros(u_shape_no_time, device=u.device)
        for i in range(b):
            idx_i = idx[i]
            if idx_i == 0 or idx_i == 1: # use one-sided fourth order richardson extrapolation 
                # Kumar Rahul, S.N. Bhattacharyya, One-sided finite-difference approximations suitable for use with Richardson extrapolation
                # http://ftp.demec.ufpr.br/CFD/bibliografia/MER/Rahul_Bhattacharyya_2006.pdf

                u_t = u[i, idx_i]
                u_tplus1 = u[i, idx_i+1]
                u_tplus2 = u[i, idx_i+2]
                u_tplus3 = u[i, idx_i+3]
                u_tplus4 = u[i, idx_i+4]
                du_dt[i] = -1 * (25 * u_t - 48 * u_tplus1 + 36 * u_tplus2 - 16 * u_tplus3 + 3 * u_tplus4) / (12 * dt)
            elif idx_i == last_idx or idx_i == last_idx - 1: # use one-sided fourth order richardson extrapolation
                u_t = u[i, idx_i]
                u_tminus1 = u[i, idx_i-1]
                u_tminus2 = u[i, idx_i-2]
                u_tminus3 = u[i, idx_i-3]
                u_tminus4 = u[i, idx_i-4]
                du_dt[i] = (25 * u_t - 48 * u_tminus1 + 36 * u_tminus2 - 16 * u_tminus3 + 3 * u_tminus4) / (12 * dt)
            else:
                u_tminus1 = u[i, idx_i-1]
                u_tminus2 = u[i, idx_i-2]
                u_tplus1 = u[i, idx_i+1]
                u_tplus2 = u[i, idx_i+2]
                du_dt[i] = 4/3 * (u_tplus1 - u_tminus1)/(2*dt) - 1/3 * (u_tplus2 - u_tminus2)/(4*dt)

    return du_dt