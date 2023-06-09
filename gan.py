from dataset_preprocessing import Dataset
import torch
import torch.nn as nn


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, L = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, L).to(device)
    interpolated_images = real * epsilon + fake * (1-epsilon)

    # calculate critic scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def gradient_penalty_cond(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, L = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, L).to(device)
    interpolated_images = real * epsilon + fake * (1-epsilon)

    mixed_scores = critic(interpolated_images, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

class Disc_dcgan_gp_1d(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv1d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),  
            nn.LeakyReLU(0.2),
           
            self._block(features_d, features_d * 2, 4, 2, 1),  
            self._block(features_d * 2, features_d * 4, 4, 2, 1), 
            self._block(features_d * 4, features_d * 8, 4, 2, 1), 
            nn.Conv1d(features_d * 8, 1, kernel_size=4, stride=2, padding=0), 
            nn.Linear(2, 1) 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False,),
            nn.InstanceNorm1d(out_channels, affine=True), 
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Gen_dcgan_gp_1d(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, filtered=False):
        super().__init__()
        self.filtered = filtered
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),  
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  
            self._block(features_g * 4, features_g * 2, 4, 2, 1), 
            nn.ConvTranspose1d(
               features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
             ),  
            nn.Linear(64, 120, bias=False),
            nn.Tanh(),  
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.filtered:
            return torch.abs(self.gen(x))
        else:
            return self.gen(x)
        

class Disc_ac_wgan_gp_1d(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super().__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv1d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),       
            self._block(features_d * 2, features_d * 4, 4, 2, 1),   
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  

            nn.Conv1d(features_d * 8, 1, kernel_size=4, stride=2, padding=0, dilation=1),
            nn.Linear(2, 1)
        )
        self.embed = nn.Embedding(num_classes, img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  
            ),
            nn.InstanceNorm1d(out_channels, affine=True),   
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        out = self.disc(x)
        return out


class Gen_ac_wgan_gp_1d(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size, filtered=False):
        super().__init__()
        self.img_size = img_size
        self.filtered = filtered
        self.gen = nn.Sequential(
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0), 
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  
            self._block(features_g * 8, features_g * 4, 4, 2, 1), 
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  
            nn.ConvTranspose1d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  
            nn.Linear(64, 120, bias=False),    
            nn.Tanh(), 
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        temp = self.embed(labels)
        embedding = temp.unsqueeze(2)
        x = torch.cat([x, embedding], dim=1)
        if self.filtered:
            return torch.abs(self.gen(x))
        else:
            return self.gen(x)
    

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

  