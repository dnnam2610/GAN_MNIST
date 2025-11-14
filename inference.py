import os
import torch
from gan import Generator
from torchvision.utils import save_image

if __name__ == '__main__':
    checkpoint_path = os.path.join("checkpoints", "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    generator = Generator()
    generator.load_state_dict(checkpoint["generator_state"])

    generator.eval()
    num_images = 50  # số ảnh muốn sinh
    z_dim = 100       # kích thước latent vector

    # Tạo folder để lưu ảnh
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(num_images):
            sample = torch.randn(1, z_dim, device='cpu')
            test_image = generator(sample).view(1, 1, 28, 28)
            test_image = (test_image + 1) / 2  # [-1,1] → [0,1]

            save_path = os.path.join(output_dir, f"sample_{i+1}.png")
            save_image(test_image, save_path, nrow=1)
            print(f"✅ Saved generated image {i+1} to {save_path}")