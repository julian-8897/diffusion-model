import torch
from torchsummary import summary

from models.simple_unet import SimpleUnet


def main():
    model = SimpleUnet()
    model.load_state_dict(torch.load("saved_models/ddpm_unet.pth", weights_only=True))
    summary(model)


if __name__ == "__main__":
    main()
