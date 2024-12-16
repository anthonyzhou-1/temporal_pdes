import lightning as L
from torch.utils.data import DataLoader

class PDEDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataconfig,) -> None:
        
        super().__init__()
        self.data_config = dataconfig
        self.dataset_config = dataconfig["dataset"]
        self.batch_size = dataconfig["batch_size"]
        self.num_workers = dataconfig["num_workers"]
        self.pde = self.dataset_config["pde"]

        if self.pde == "burgers_2d" or self.pde == "ns_2d": 
            from dataset.dataset_2D import PDEDataset2D
            self.train_dataset = PDEDataset2D(path = self.dataset_config["train_path"],
                                                pde = self.dataset_config["pde"],
                                                split = "train",
                                                resolution = self.dataset_config["resolution"],
                                                start = self.dataset_config["start"] if "start" in self.dataset_config else 0.0)
            self.val_dataset = PDEDataset2D(path = self.dataset_config["valid_path"],
                                            pde = self.dataset_config["pde"],
                                            split = "valid",
                                            resolution = self.dataset_config["resolution"],
                                            start = self.dataset_config["start"] if "start" in self.dataset_config else 0.0)
        elif self.pde == "cylinder_2d":
            from dataset.cylinder import CylinderMeshDataset
            from dataset.normalizer import Normalizer
            normalizer_config = self.data_config["normalizer"]
            self.train_dataset = CylinderMeshDataset(path=self.dataset_config["train_path"],
                                                        time_horizon=self.dataset_config["time_horizon"],
                                                        time_start=self.dataset_config["time_start"],
                                                        idx_path=self.dataset_config["train_idx_path"])
            self.val_dataset = CylinderMeshDataset(path=self.dataset_config["valid_path"],
                                                        time_horizon=self.dataset_config["time_horizon"],
                                                        time_start=self.dataset_config["time_start"],
                                                        idx_path=self.dataset_config["valid_idx_path"])
            self.normalizer = Normalizer(dataset=self.train_dataset,
                                        **normalizer_config)
        else:
            from dataset.dataset_1D import PDEDataset1D
            self.train_dataset = PDEDataset1D(path=self.dataset_config["train_path"],
                                            pde=self.dataset_config["pde"],
                                            split="train",
                                            resolution=self.dataset_config["resolution"],)
            self.val_dataset = PDEDataset1D(path=self.dataset_config["valid_path"],
                                            pde=self.dataset_config["pde"],
                                            split="valid",
                                            resolution=self.dataset_config["resolution"],)

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # Eager imports to avoid specific dependencies that are not needed in most cases

        if stage == "fit":
            pass 

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        self.pin_memory = False if self.num_workers == 0 else True
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None
