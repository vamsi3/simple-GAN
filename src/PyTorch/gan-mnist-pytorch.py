import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Importing torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# For MNIST dataset and visualization
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Getting the command-line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help="number of epochs to train")
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay rate of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay rate of second order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--output_dir', type=str, default='output', help='name of output directory')
args = parser.parse_args()

img_shape = (args.channels, args.img_size, args.img_size)

# Check CUDA's presence
cuda_is_present = True if torch.cuda.is_available() else False

class Generator(nn.Module):
	def __init__(self):
		super().__init__()

		def layer_block(input_size, output_size, normalize=True):
			layers = [nn.Linear(input_size, output_size)]
			if normalize:
				layers.append(nn.BatchNorm1d(output_size, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*layer_block(args.latent_dim, 128, normalize=False),
			*layer_block(128, 256),
			*layer_block(256, 512),
			*layer_block(512, 1024),
			nn.Linear(1024, int(np.prod(img_shape))),
			nn.Tanh()
		)

	def forward(self, z):
		img = self.model(z)
		img = img.view(img.size(0), *img_shape)
		return img

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, img):
		img_flat = img.view(img.size(0), -1)
		verdict = self.model(img_flat)
		return verdict

# Utilize CUDA if available
generator = Generator()
discriminator = Discriminator()
adversarial_loss = torch.nn.BCELoss()

if cuda_is_present:
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()

# Loading MNIST dataset
os.makedirs('data/mnist', exist_ok=True)
data_loader = torch.utils.data.DataLoader(
	datasets.MNIST('/data/mnist', train=True, download=True,
		transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])),
	batch_size=args.batch_size, shuffle=True)

# Training the GAN
os.makedirs(f'{args.output_dir}/images', exist_ok=True)
Tensor = torch.cuda.FloatTensor if cuda_is_present else torch.FloatTensor

optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))

losses = []
images_for_gif = []
for epoch in range(1, args.epochs+1):
	for i, (images, _) in enumerate(data_loader):

		real_images = Variable(images.type(Tensor))
		real_output = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
		fake_output = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)

		# Training Generator
		optimizer_generator.zero_grad()
		z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], args.latent_dim))))
		generated_images = generator(z)
		generator_loss = adversarial_loss(discriminator(generated_images), real_output)
		generator_loss.backward()
		optimizer_generator.step()

		# Training Discriminator
		optimizer_discriminator.zero_grad()
		discriminator_loss_real = adversarial_loss(discriminator(real_images), real_output)
		discriminator_loss_fake = adversarial_loss(discriminator(generated_images.detach()), fake_output)
		discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2
		discriminator_loss.backward()
		optimizer_discriminator.step()

		print(f"[Epoch {epoch:=4d}/{args.epochs}] [Batch {i:=4d}/{len(data_loader)}] ---> "
			f"[D Loss: {discriminator_loss.item():.6f}] [G Loss: {generator_loss.item():.6f}]")

	losses.append((generator_loss.item(), discriminator_loss.item()))
	image_filename = f'{args.output_dir}/images/{epoch}.png'
	save_image(generated_images.data[:25], image_filename, nrow=5, normalize=True)
	images_for_gif.append(imageio.imread(image_filename))

# Visualizing the losses at every epoch
losses = np.array(losses)
plt.plot(losses.T[0], label='Generator')
plt.plot(losses.T[1], label='Discriminator')
plt.title("Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'{args.output_dir}/loss_plot.png')

# Creating a gif of generated images at every epoch
imageio.mimwrite(f'{args.output_dir}/generated_images.gif', images_for_gif, fps=len(images)/5)
