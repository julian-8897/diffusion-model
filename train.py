import torch
import torch.nn as nn
from tqdm import tqdm

from data.data_loader import Flowers102Dataloader
from models.simple_unet import SimpleUnet
from utils import get_loss, sample_plot_image


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


def train_loop(dataloader, model, optimizer, batch_size, epochs, T, device):
    model.to(device)
    model.train()
    scaler = torch.GradScaler() if device == "cuda" else None

    for epoch in range(epochs):
        with tqdm(
            total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        ) as pbar:

            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                if device == "cuda":
                    with torch.autocast(device_type=device):
                        t = torch.randint(0, T, (batch_size,), device=device).long()
                        loss = get_loss(model, batch[0], t, device)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    t = torch.randint(0, T, (batch_size,), device=device).long()
                    loss = get_loss(model, batch[0], t, device)
                    loss.backward()
                    optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image(model, dataloader.img_size, device)


def main():
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current device: ", device)
    dataloader = Flowers102Dataloader(img_size=64, batch_size=64).dataloader
    model = SimpleUnet()
    model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 1
    T = 300

    train_loop(dataloader, model, optimizer, dataloader.batch_size, epochs, T, device)
    save_model(model, "saved_models/ddpm_unet.pth")


if __name__ == "__main__":
    main()
