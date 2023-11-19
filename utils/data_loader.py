import json

from monai import data
from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    LoadImaged,
    NormalizeIntensityd,
    RandSpatialCropd,
)
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, batch_size, dataset_path, dataset_json_file_path):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        trainingData, testingData = self.read_data(dataset_json_file_path)
        trainingData, validationData = self.split_data(trainingData)

        self.validationData = self.update_paths(validationData, isTraining=True)
        self.testingData = self.update_paths(testingData, isTraining=False)
        self.trainingData = self.update_paths(trainingData, isTraining=True)
        (
            self.train_loader,
            self.val_loader,
            self.train_dataset_length,
            self.validation_dataset_length,
        ) = self.data_loader()

    def read_data(self, path):
        with open(path) as json_file:
            dataset = json.load(json_file)

        return dataset["training"], dataset["testing"]

    def split_data(self, dataLst, test_size=0.15):
        trainingPaths, validationPaths = train_test_split(
            dataLst, test_size=test_size, random_state=42
        )
        return trainingPaths, validationPaths

    def update_paths(self, dataset, isTraining=True):
        for i in range(len(dataset)):
            for j in range(len(dataset[i]["images"])):
                dataset[i]["images"][j] = (
                    self.dataset_path + "/" + dataset[i]["images"][j]
                )
            if isTraining:
                dataset[i]["mask"] = self.dataset_path + "/" + dataset[i]["mask"]

        return dataset

    def data_loader(self):
        transform = Compose(
            [
                LoadImaged(keys=["images", "mask"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
                RandSpatialCropd(
                    keys=["images", "mask"], roi_size=[128, 128, 64], random_size=False
                ),
                NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
            ]
        )

        train_dataset = data.Dataset(data=self.trainingData, transform=transform)
        validation_dataset = data.Dataset(data=self.validationData, transform=transform)

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = data.DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        return train_loader, val_loader, len(train_dataset), len(validation_dataset)
