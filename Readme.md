# Optical Flow Estimation 

This project focuses on the [Optical Flow computer vision task](https://paperswithcode.com/task/optical-flow-estimation), specifically testing various libraries and state-of-the-art (SOTA) models using C++.

The initial focus is on implementing and testing the [RAFT](https://github.com/princeton-vl/RAFT) model (Recurrent All-Pairs Field Transforms).

## RAFT Optical Flow with TorchScript and C++

This section details the process of exporting, loading, and performing inference with RAFT optical flow models using TorchScript in a C++ environment.

### Dependencies

*   **Libtorch:** Version [2.5.1](https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip) (CUDA 12.4 build, adjust if needed).
*   **Reference:** [PyTorch RAFT Documentation](https://pytorch.org/vision/main/models/raft.html)

**Note:** The provided Libtorch link is for a CUDA 12.4 build. If you are using a different CUDA version or a CPU-only build, you will need to download the appropriate Libtorch version from the [PyTorch website](https://pytorch.org/get-started/locally/).

### Model Export
[RAFT Model Export](docs/raft_model_export.md)

## Feedback
This project is under active development. Contributions and suggestions are welcome.