import torch

from data.data_loader import Flowers102Dataloader
from models.simple_unet import SimpleUnet
from utils import get_loss, sample_plot_image


def train_loop(dataloader, model, optimizer, batch_size, epochs, T, device):
    model.to(device)
    model.train()
    scaler = torch.GradScaler() if device == "cuda" else None

    for epoch in range(epochs):
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

        if epoch % 1 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            sample_plot_image(model, dataloader.img_size, device)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Current device: ", device)
    dataloader = Flowers102Dataloader(img_size=64, batch_size=1).dataloader
    model = SimpleUnet()
    model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100
    T = 300

    train_loop(dataloader, model, optimizer, dataloader.batch_size, epochs, T, device)


if __name__ == "__main__":
    main()
