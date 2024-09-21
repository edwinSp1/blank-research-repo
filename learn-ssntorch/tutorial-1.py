import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
from snntorch import spikegen

TRAINING_SUBSET = 10
# Training Parameters
batch_size=128
data_path='/tmp/data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Torch Variables
dtype = torch.float

transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
# cut the dataset size cuz we aren't traning yet :)
mnist_train = utils.data_subset(mnist_train, TRAINING_SUBSET)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)


# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

num_steps = 100
# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)
spike_data_sample = spike_data[:, 0, 0]

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
# plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

HTML(anim.to_html5_video())