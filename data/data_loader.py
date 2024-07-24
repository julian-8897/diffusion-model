import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class Flowers102Dataloader:
    def __init__(self, img_size, batch_size):
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform_data = self._create_transforms()
        self.dataset = self._load_transformed_dataset()

    def _create_transforms(self):
        data_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Scales data into [0,1]
            transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ]
        return transforms.Compose(data_transforms)

    def _load_transformed_dataset(self):
        train = torchvision.datasets.Flowers102(
            root=".", download=True, transform=self.transform_data
        )

        test = torchvision.datasets.Flowers102(
            root=".", download=True, transform=self.transform_data, split="test"
        )
        return torch.utils.data.ConcatDataset([train, test])

    @property
    def dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )


def main():
    IMG_SIZE = 64
    BATCH_SIZE = 32
    dataloader = Flowers102Dataloader(IMG_SIZE, BATCH_SIZE).dataloader

    print(f"Number of batches: {len(dataloader)}")
    print(f"Total number of samples: {len(dataloader.dataset)}")


if __name__ == "__main__":
    main()
