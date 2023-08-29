import json
from torch.utils import data
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImaged, ConvertToMultiChannelBasedOnBratsClassesd, RandSpatialCropd, NormalizeIntensityd)


class Dataset:

    def __init__(self, batch_size, json_path):
        self.batch_size = batch_size
        self.trainingData, self.testData = self.read_data(json_path)
        self.trainingData, self.validationData = self.split_data(
            self.trainingData)
        self.train_loader, self.val_loader, self.train_dataset_length, self.validation_dataset_length = self.data_loader()

    def read_data(path):
        with open(path) as json_file:
            dataset = json.load(json_file)

        return dataset['training'], dataset['validation']

    def split_data(dataLst, test_size=0.15):
        trainingPaths, validationPaths = train_test_split(
            dataLst, test_size=test_size, random_state=42)
        return trainingPaths, validationPaths

    def data_loader(self):
        trainTransform = Compose(
            [
                LoadImaged(keys=["images", "mask"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
                RandSpatialCropd(keys=["images", "mask"], roi_size=[
                                 128, 128, 64], random_size=False),
                NormalizeIntensityd(
                    keys="images", nonzero=True, channel_wise=True),
            ]
        )

        valTransform = Compose(
            [
                LoadImaged(keys=["images", "mask"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"),
                RandSpatialCropd(keys=["images", "mask"], roi_size=[
                                 128, 128, 64], random_size=False),
                NormalizeIntensityd(
                    keys="images", nonzero=True, channel_wise=True),
            ]
        )

        train_dataset = data.Dataset(
            data=self.trainingData, transform=trainTransform)
        validation_dataset = data.Dataset(
            data=self.validationData, transform=valTransform)

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
