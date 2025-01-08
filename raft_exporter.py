import torch
from torchvision.models.optical_flow import raft_small, raft_large
import torch.nn as nn
import argparse


def load_model(model_type='small'):
    """Load and prepare RAFT model for export."""
    # Select model based on type
    model_fn = raft_small if model_type == 'small' else raft_large
    model = model_fn(pretrained=True)
    
    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, device


def prepare_sample_input(batch_size, height, width, device):
    """Create sample input tensors for model export."""
    input1 = torch.randn(batch_size, 3, height, width).to(device)
    input2 = torch.randn(batch_size, 3, height, width).to(device)
    return input1, input2


def export_torchscript_trace(model, example_inputs, model_type, output_dir, dynamic=True):
    """Export model using TorchScript tracing."""
    print(f"Exporting {model_type} model to TorchScript (Tracing)...")
    try:
        # Create wrapper for dynamic support
        class DynamicRAFT(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x1, x2):
                return self.model(x1, x2)

        if dynamic:
            wrapped_model = DynamicRAFT(model)
            traced_model = torch.jit.trace(
                wrapped_model,
                example_inputs,
                check_trace=True,
                check_tolerance=1e-4
            )
            filename = f"{output_dir}/raft_{model_type}_traced_dynamic.pt"
            traced_model.save(filename, _extra_files={
                "dynamic_axes.json": str({
                    "input1": {0: "batch"},
                    "input2": {0: "batch"},
                    "output": {0: "batch"}
                })
            })
            print(f"Dynamic traced TorchScript model saved as '{filename}'")
        else:
            # Original static tracing
            traced_model = torch.jit.trace(model, example_inputs)
            filename = f"{output_dir}/raft_{model_type}_traced_torchscript.pt"
            traced_model.save(filename)
            print(f"Static traced TorchScript model saved as '{filename}'")
    except Exception as e:
        print(f"Tracing failed: {e}")


def export_torchscript_script(model, model_type, output_dir):
    """Export model using TorchScript scripting."""
    print(f"Exporting {model_type} model to TorchScript (Scripting)...")
    try:
        scripted_model = torch.jit.script(model)
        filename = f"{output_dir}/raft_{model_type}_scripted_torchscript.pt"
        scripted_model.save(filename)
        print(f"Scripted TorchScript model saved as '{filename}'")
    except Exception as e:
        print(f"Scripting failed: {e}")


def export_onnx(model, example_inputs, model_type, output_dir):
    """Export model to ONNX format."""
    print(f"Exporting {model_type} model to ONNX...")
    try:
        filename = f"{output_dir}/raft_{model_type}.onnx"
        input1, input2 = example_inputs
        
        torch.onnx.export(
            model,
            (input1, input2),
            filename,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['input1', 'input2'],
            output_names=['output'],
            dynamic_axes={
                'input1': {0: 'batch_size', 2: 'height', 3: 'width'},
                'input2': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print(f"ONNX model saved as '{filename}'")
    except Exception as e:
        print(f"ONNX export failed: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export RAFT optical flow models to various formats.')
    
    parser.add_argument('--model-type', type=str, choices=['small', 'large', 'both'],
                      default='both', help='Type of RAFT model to export (default: both)')
    
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for sample inputs (default: 1)')
    
    parser.add_argument('--height', type=int, default=520,
                      help='Height of sample inputs (default: 520)')
    
    parser.add_argument('--width', type=int, default=960,
                      help='Width of sample inputs (default: 960)')
    
    parser.add_argument('--output-dir', type=str, default='.',
                      help='Output directory for exported models (default: current directory)')
    
    parser.add_argument('--format', type=str, 
                      choices=['all', 'traced', 'scripted', 'onnx'],
                      default='all',
                      help='Export format(s) to use (default: all)')
    
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                      default=None,
                      help='Device to use (default: use cuda if available)')

    parser.add_argument('--dynamic', action='store_true',
                      help='Enable dynamic batching for traced export')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine which models to export
    model_types = ['small', 'large'] if args.model_type == 'both' else [args.model_type]
    
    # Process each model type
    for model_type in model_types:
        print(f"\nProcessing RAFT {model_type} model...")
        
        # Load model
        model, _ = load_model(model_type)
        model = model.to(device)
        
        # Prepare sample inputs
        example_inputs = prepare_sample_input(args.batch_size, args.height, args.width, device)
        
        # Export in specified format(s)
        if args.format in ['all', 'traced']:
            export_torchscript_trace(model, example_inputs, model_type, args.output_dir, dynamic=args.dynamic)
        
        if args.format in ['all', 'scripted']:
            export_torchscript_script(model, model_type, args.output_dir)
        
        if args.format in ['all', 'onnx']:
            export_onnx(model, example_inputs, model_type, args.output_dir)


if __name__ == "__main__":
    main()