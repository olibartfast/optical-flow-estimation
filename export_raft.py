
import torch
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchsummary import summary

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()


# traced_model = torch.jit.script(model)
# traced_model.save("raft_large.pt")

batch_size = 1
channels = 3
height = 384
width = 512

# Create dummy inputs (two images, image1 and image2)
dummy_input1 = torch.randn(batch_size, channels, height, width)
dummy_input2 = torch.randn(batch_size, channels, height, width)

# Print the summary to get the size of the output layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, [(channels, height, width), (channels, height, width)])  # Pass the dummy inputs as a list