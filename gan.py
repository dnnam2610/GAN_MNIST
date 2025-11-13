import torch.nn as nn
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super().__init__()
        self.ln1 = self.make_linear_block(img_size*img_size, 1024)
        self.ln2 = self.make_linear_block(1024, 512)
        self.ln3 = self.make_linear_block(512, 256)
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def make_linear_block(self, input_chanels, output_chanels, drop_rate=0.3):
        return nn.Sequential(
            nn.Linear(in_features=input_chanels, out_features=output_chanels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=drop_rate)
        )
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.out(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.ln1 = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
        )
        self.ln2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )
        self.ln3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(
            nn.Linear(1024, img_size * img_size),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.out(x)
        return x
    
if __name__ == '__main__':
    generator = Generator(latent_dim=100, img_size=28)
    discriminator = Discriminator(img_size=28)

    summary(generator, (100,))     # Generator input: noise vector 100-dim
    summary(discriminator, (1, 28, 28))