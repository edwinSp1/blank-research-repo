from model import *

input_tensor = torch.tensor(
    [
        1, 1, 1,
        0, 1, 0, 
        0, 0, 0
    ], dtype=torch.float
).to(device)

print(net(
  input_tensor  
))
from snntorch import spikegen

# Spiking Data
spike_data = spikegen.rate(input_tensor, num_steps=num_steps)

print(spike_data)