# Algorithm-Hardware Co-Design: CNN Optimization & C++ Memory Tuning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Computer Architecture](https://img.shields.io/badge/Domain-Computer_Architecture-blue?style=for-the-badge)

## 📌 Overview

This project demonstrates a complete **algorithm-hardware co-design** workflow, bridging the gap from high-level neural network architecture design to low-level C++ memory access optimization. The primary objective is to maximize computational throughput and reduce memory bandwidth pressure—core challenges in modern AI accelerator chip design.

## 🚀 Technical Deep Dive & What I Learned

### 1. CNN & Computer Vision Foundations
* **Low-Level Filter Implementation:** Rather than relying on high-level wrappers, implemented classic image processing filters (Average, Gaussian, Laplacian, Edge Detection) from scratch using PyTorch tensor operations and `conv2d`.
* **Model Construction & Training Loop:** Built a LeNet-5 style Convolutional Neural Network for CIFAR-10 image classification. Fully engineered the optimization pipeline, including the forward pass, Cross-Entropy Loss computation, backpropagation, and evaluation inference using `torch.no_grad()`.
* **Math & Gradient Derivation:** Deepened the theoretical understanding of combining Softmax with Cross-Entropy Loss, successfully deriving its elegantly simple gradient formula ($dL/dZ = P - Y$).

### 2. Algorithm-Level Hardware Optimization (Filter Decomposition)
* **Reducing Computational Complexity:** Implemented **filter decomposition** techniques to break down a large $5\times5$ convolutional kernel into two $3\times3$ kernels, and further into **spatially separable convolutions** ($5\times1$ and $1\times5$).
* **Hardware Awareness:** Proved that this algorithm-level decomposition maintains the same Receptive Field while drastically reducing the number of model parameters and Multiply-Accumulate (MAC) operations. This is crucial for easing memory bandwidth pressure and reducing power consumption in AI accelerators.

### 3. C++ Matrix Operations & Memory Hierarchy Tuning
* **Spatial Locality & Hardware Prefetching:** In C++ Matrix Multiplication (GEMM) benchmarks, demonstrated the massive performance impact of "Row-Major" memory storage. Verified that the `kij` loop order vastly outperforms the intuitive `ijk` order due to perfect alignment with the hardware prefetcher's contiguous memory access patterns.
* **Cache Tiling / Blocking:** Resolved Cache Thrashing caused by large matrix operations by implementing Tiling to improve Temporal Locality. Through empirical experimentation, identified the optimal Tile Size (e.g., $T=64$) that perfectly fits the working blocks into the L1 Data Cache, maximizing CPU computation throughput.

---

## 📂 Project Structure

```text
├── src/
│   ├── cpp/                 # C++ implementation for GEMM and Cache Tiling
│   └── python/              # PyTorch CNN and filter decomposition scripts
├── notebooks/               # Jupyter notebooks for math derivation and visualization
├── results/                 # Benchmark logs and performance graphs
└── README.md
