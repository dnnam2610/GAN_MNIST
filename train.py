import os
import numpy as np
import torch
from gan import Discriminator, Generator
from torch import optim
import torch.nn as nn

from dataset import load_mnist
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

def train_discriminator(disc, optimizer, loss_func, real_data, fake_data, device):
    optimizer.zero_grad()
    pred_real = disc(real_data)
    loss_real = loss_func(pred_real, torch.ones(real_data.size(0), 1, device=device))
    loss_real.backward()

    pred_fake = disc(fake_data)
    loss_fake = loss_func(pred_fake, torch.zeros(fake_data.size(0), 1, device=device))
    loss_fake.backward()

    optimizer.step()

    return loss_real + loss_fake, pred_real, pred_fake


def train_generator(disc, optimizer, loss_func, fake_data, device):
    optimizer.zero_grad()
    pred = disc(fake_data)
    loss = loss_func(pred, torch.ones(fake_data.size(0), 1, device=device))
    loss.backward()

    optimizer.step()

    return loss

def save_checkpoint(state, checkpoint_path, filename):
    filepath = os.path.join(checkpoint_path, filename)
    torch.save(state, filepath)
    print(f"✅ Saved checkpoint: {filepath}")


def load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer):
    latest_ckpt = os.path.join(checkpoint_path, "last.pt")
    if os.path.isfile(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location="cpu")
        generator.load_state_dict(checkpoint["generator_state"])
        discriminator.load_state_dict(checkpoint["discriminator_state"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_g_loss = checkpoint.get("best_g_loss", float("inf"))
        print(f"🔁 Resumed from checkpoint at epoch {checkpoint['epoch']}")
        return start_epoch, best_g_loss
    else:
        print("⚠️ No checkpoint found — starting from scratch.")
        return 0, float("inf")

def train(epochs=100, batch_size=64, noise_dim=100, learning_rate=0.0002, log_path="logs", checkpoint_path="checkpoints", img_path="img",  resume_training=False):

    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    if not os.path.isdir(img_path):
        os.makedirs(img_path)

    writer = SummaryWriter(log_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    train_data = load_mnist()
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )

    discriminator = Discriminator(img_size=28).to(device)
    generator = Generator(latent_dim=100, img_size=28).to(device)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate) 

    start_epoch, best_g_loss = (0, float("inf"))
    if resume_training:
        start_epoch, best_g_loss = load_checkpoint(
            checkpoint_path, generator, discriminator, g_optimizer, d_optimizer
        )

    num_samples = 16
    test_noise = torch.randn(num_samples, noise_dim, device=device)

    for epoch in range(start_epoch, epochs):
        discriminator.train()
        generator.train()
        progress_bar = tqdm(train_dataloader, colour="cyan", desc=f"Training epoch {epoch+1}/{epochs}")
        total_disc_loss = []
        total_gen_loss = []
        for batch_idx, (batch, _) in enumerate(progress_bar):

            # Train DISCRIMINATOR
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise).detach()
            loss_disc, pred_real, pred_fake = train_discriminator(
                disc=discriminator,
                optimizer=d_optimizer,
                loss_func= nn.BCELoss(),
                real_data=batch.to(device),
                fake_data= fake_data,
                device=device
            )

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)

            loss_gen = train_generator(
                disc=discriminator,
                optimizer=g_optimizer,
                loss_func= nn.BCELoss(),
                fake_data=fake_data,
                device=device
            )


            total_disc_loss.append(loss_disc.item())
            total_gen_loss.append(loss_gen.item())
        
            avg_disc_loss =  np.mean(total_disc_loss)
            avg_gen_loss = np.mean(total_gen_loss)
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Loss Disc: {avg_disc_loss:.2f} | Loss Gen: {avg_gen_loss:.2f}")
            writer.add_scalar("Loss_disc", avg_disc_loss, global_step=batch_idx+epoch*len(train_dataloader))
            writer.add_scalar("Loss_gen", avg_gen_loss, global_step=batch_idx+epoch*len(train_dataloader))

        # === Save last img ===
        if epoch in [0, 299, 599, 1199, 2399]:
            generator.eval()
            with torch.no_grad():
                test_images = generator(test_noise).view(num_samples, 1, 28, 28)
                test_images = (test_images + 1) / 2  # chuyển từ [-1,1] → [0,1] để hiển thị ảnh đúng
                save_path = os.path.join(img_path, f"epoch_{epoch+1}.png")
                save_image(test_images, save_path, nrow=8)
                print(f"✅ Saved generated samples to {save_path}")
        # === Save last checkpoint ===
        save_checkpoint(
            {
                "epoch": epoch,
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "g_optimizer_state": g_optimizer.state_dict(),
                "d_optimizer_state": d_optimizer.state_dict(),
                "best_g_loss": best_g_loss,
            },
            checkpoint_path,
            "last.pt",
        )

        # === Save best checkpoint ===
        if avg_gen_loss < best_g_loss:
            best_g_loss = avg_gen_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "g_optimizer_state": g_optimizer.state_dict(),
                    "d_optimizer_state": d_optimizer.state_dict(),
                    "best_g_loss": best_g_loss,
                },
                checkpoint_path,
                "best.pt",
            )
            print(f"🌟 New best model saved at epoch {epoch+1} with G loss = {best_g_loss:.4f}")

    print("✅ Training completed!")
if __name__ == '__main__':
    train(
        epochs=2400,
        batch_size=128,
        log_path='logs',
        checkpoint_path='checkpoints',
        img_path="img",
        resume_training=False
    )