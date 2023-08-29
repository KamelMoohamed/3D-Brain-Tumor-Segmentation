from utils import Dataset
from models import VAEModel
import torch


if __name__ == "__main__":
    dataset = Dataset(
        batch_size=32, json_path='/content/drive/MyDrive/Brain_tumor_segmentation/brain_dataset.json')

    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    train_dataset_length = dataset.train_dataset_length
    validation_dataset_length = dataset.validation_dataset_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    model = VAEModel(device, train_loader, val_loader, epochs)

    model.train()
