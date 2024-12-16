import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri as mtri

def plot_result(traj, traj_pred, path):
    # result is a trajectory in shape (b, nt, nx)
    idx = 0
    u = traj[idx].detach().cpu().numpy() # shape (nt, nx)
    u_pred = traj_pred[idx].detach().cpu().numpy()
    x = np.linspace(0, 2, u.shape[1])
    
    interval = u.shape[0] // 5

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(0, u.shape[0], interval):
        axs[0].plot(x, u[i], label='t={}'.format(i))
        axs[1].plot(x, u_pred[i], label='t={}'.format(i))
    
    axs[0].set_title('True trajectory')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('u')
    axs[0].legend()

    axs[1].set_title('Predicted trajectory')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('u')
    axs[1].legend()

    plt.savefig(path)
    plt.close()
    return 

def plot_result_2d(u, rec, n_t=5, path=None):
    u = u[0].detach().cpu()
    rec = rec[0].detach().cpu() if rec is not None else None
    
    if len(u.shape) > 3:
        u = u[..., 0] # get the first channel
        rec = rec[..., 0] if rec is not None else None
        
    # u in shape nt nx ny 
        
    vmin = torch.min(u)
    vmax = torch.max(u)

    n_skip = u.shape[0] // n_t 
    u_downs = u[::n_skip]

    if rec is not None:
        rec_downs = rec[::n_skip]
        fig, ax = plt.subplots(n_t, 2, figsize=(8, 4*n_t))
        for j in range(2):
            for i in range(n_t):
                ax[i][j].set_axis_off()
                if j == 0:
                    velocity = u_downs[i] 
                else:
                    velocity = rec_downs[i]

                im = ax[i][j].imshow(velocity, vmin=vmin, vmax=vmax, cmap='inferno')
                ax[i][j].title.set_text(f'Timestep {i*n_skip}')
            ax[0][j].title.set_text(f'Ground Truth' if j == 0 else f'Prediction')
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))

        for i in range(n_t):
            ax[i].set_axis_off()
            velocity = u_downs[i] 

            im = ax[i].imshow(velocity, vmin=vmin, vmax=vmax, cmap='inferno')
            ax[i].title.set_text(f'Timestep {i*n_skip}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)

def plot_result_mesh(u, mesh_pos, cells, n_t=5, rec = None, path=None):
    u = u[0].detach().cpu()
    rec = rec[0].detach().cpu() if rec is not None else None
    mesh_pos = mesh_pos[0].detach().cpu()
    cells = cells[0].detach().cpu()
    
    if len(u.shape) > 2:
        u = u[..., 0] # get the first channel
        rec = rec[..., 0] if rec is not None else None

    # u in shape nt m
    # mesh_pos in shape m, 2
    # cells in shape n_edges, 3
    # plots time-dependent mesh data at n_t timesteps

    vmin = torch.min(u)
    vmax = torch.max(u)

    n_skip = u.shape[0] // n_t 
    u_downs = u[::n_skip]

    if rec is not None:
        fig, ax = plt.subplots(n_t, 2, figsize=(12, 2*n_t))
        rec_downs = rec[::n_skip]
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(6, 2*n_t))

    if rec is None:
        for i in range(n_t):
            ax[i].set_axis_off()
            pos = mesh_pos
            faces = cells
            velocity = u_downs[i]
            triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
            tpc = ax[i].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[i].triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax[i].title.set_text(f'Timestep {i*n_skip}')
    else:
        for i in range(n_t):
            ax[i][0].set_axis_off()
            pos = mesh_pos
            faces = cells
            velocity = u_downs[i]
            triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
            tpc = ax[i][0].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[i][0].triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax[i][0].title.set_text(f'Ground Truth')

            ax[i][1].set_axis_off()
            pos = mesh_pos
            faces = cells
            velocity = rec_downs[i]
            triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
            tpc = ax[i][1].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
            ax[i][1].triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax[i][1].title.set_text(f'Pred')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(tpc, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)

def plot_mesh(u, mesh_pos, cells, rec = None, path=None):
    # u in shape b m 
    # mesh_pos in shape b m 2
    # cells in shape b n_edges 3

    u = u[0].detach().cpu()
    rec = rec[0].detach().cpu() if rec is not None else None
    mesh_pos = mesh_pos[0].detach().cpu()
    cells = cells[0].detach().cpu()
    
    if len(u.shape) > 1:
        u = u[..., 0] # get the first channel
        rec = rec[..., 0] if rec is not None else None

    # u in shape m
    # mesh_pos in shape m, 2
    # cells in shape n_edges, 3
    # plots time-dependent mesh data at n_t timesteps

    vmin = torch.min(u)
    vmax = torch.max(u)

    n_t = 1

    if rec is not None:
        fig, ax = plt.subplots(n_t, 2, figsize=(12, 2*n_t))
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(6, 2*n_t))

    if rec is None:
        ax.set_axis_off()
        pos = mesh_pos
        faces = cells
        velocity = u
        triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
        tpc = ax.tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
        ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    else:
        ax[0].set_axis_off()
        pos = mesh_pos
        faces = cells
        velocity = u
        triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
        tpc = ax[0].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[0].triplot(triang, 'ko-', ms=0.5, lw=0.3)
        ax[0].title.set_text(f'Ground Truth')

        ax[1].set_axis_off()
        pos = mesh_pos
        faces = cells
        velocity = rec
        triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)
        tpc = ax[1].tripcolor(triang, velocity, vmin=vmin, vmax=vmax, cmap='viridis')
        ax[1].triplot(triang, 'ko-', ms=0.5, lw=0.3)
        ax[1].title.set_text(f'Pred')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(tpc, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)