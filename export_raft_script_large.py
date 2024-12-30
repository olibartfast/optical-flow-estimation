import torch
from torchvision.models.optical_flow import raft_large  
from torchvision.transforms import functional as F

# Load the pretrained RAFT model
model = raft_large(pretrained=True).eval()

# Sample input tensors with batch size 1 and image size 256x256 (modify as needed)
example_input1 = torch.randn(1, 3, 256, 256)
example_input2 = torch.randn(1, 3, 256, 256)

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
