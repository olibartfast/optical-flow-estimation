import torch
from torchvision.models.optical_flow import raft_large
import torch.nn as nn


# Load the pretrained RAFT model
model = raft_large(pretrained=True)

# Move the model to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Batch size and image size
batch_size = 1
height = 520
width = 960

# Sample input tensors
example_input1 = torch.randn(batch_size, 3, height, width).to(device)
example_input2 = torch.randn(batch_size, 3, height, width).to(device)

# Method 1: Export to TorchScript using tracing
print("Exporting to TorchScript (Tracing)...")
traced_model = torch.jit.trace(model, (example_input1, example_input2))
traced_model.save("raft_large_traced_torchscript.pt")
print("Traced TorchScript model saved as 'raft_large_traced_torchscript.pt'")

# Method 2: Export to TorchScript using scripting (if scripting is compatible)
print("Exporting to TorchScript (Scripting)...")
try:
    scripted_model = torch.jit.script(model)
    scripted_model.save("raft_large_scripted_torchscript.pt")
    print("Scripted TorchScript model saved as 'raft_large_scripted_torchscript.pt'")
except Exception as e:
    print(f"Scripting failed: {e}")

# Method 3: Export to ONNX
print("Exporting to ONNX...")
try:
    torch.onnx.export(
        model,
        (example_input1, example_input2),
        "raft_large.onnx",
        export_params=True,
        opset_version=16,  # Choose ONNX opset version based on compatibility
        do_constant_folding=True,
        input_names=['input1', 'input2'],
        output_names=['output'],
        dynamic_axes={
            'input1': {0: 'batch_size', 2: 'height', 3: 'width'},
            'input2': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print("ONNX model saved as 'raft_large.onnx'")
except Exception as e:
    print(f"ONNX export failed: {e}")