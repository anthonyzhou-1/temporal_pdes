import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from common.plotting import plot_result, plot_result_2d, plot_result_mesh

class PlottingCallback(Callback):
    def on_validation_end(self, trainer, pl_module) -> None:
        try:        
            valid_loader = trainer.val_dataloaders
            train_mode = pl_module.train_mode
            inference_mode = pl_module.inference_mode

            with torch.no_grad():
                batch = next(iter(valid_loader))
                batch = {k: v.to(pl_module.device) for k, v in batch.items()}

                accumulated_loss, correlation_time, result, result_pred = pl_module.validation_step(batch, 0, eval=True)

                plt.plot(accumulated_loss, label='Avg Validation Loss')
                plt.axvline(correlation_time, color='r', linestyle='--', label='Steps to 0.8 correlation')
                plt.xlabel('Time step')
                plt.ylabel('Validation Loss')
                plt.legend()
                plt.title(f"Mode: {train_mode}, {inference_mode}, Correlation time: {str(correlation_time)}")
                epoch = trainer.current_epoch
                path = trainer.default_root_dir + "/error_epoch-" + str(epoch) + ".png"
                
                plt.savefig(path)
                plt.close()

                if result is not None:
                    path = trainer.default_root_dir + "/traj_epoch-" + str(epoch) + ".png"

                    if pl_module.model_name == "gnn" or pl_module.model_name == "gino" or pl_module.model_name == "oformer":
                        plot_result_mesh(result, mesh_pos= batch['pos'], cells = batch['cells'], rec=result_pred, path=path)
                    elif len(result.shape) == 3:
                        plot_result(result, result_pred, path)
                    else:
                        plot_result_2d(result, result_pred, path=path)
        except:
            print("Error in plotting")
            pass