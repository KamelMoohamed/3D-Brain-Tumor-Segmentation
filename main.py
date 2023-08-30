from utils.data_loader import Dataset
from models.vae_model import VAEModel
import torch


if __name__ == "__main__":
    dataset = Dataset(
        batch_size=5, json_path='/content/drive/MyDrive/3D Brain Tumor Segmentation/dataset.json')

    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    train_dataset_length = dataset.train_dataset_length
    validation_dataset_length = dataset.validation_dataset_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    model = VAEModel(device, train_loader, val_loader, epochs)

    model.train()
