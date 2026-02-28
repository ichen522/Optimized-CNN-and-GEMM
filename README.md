# Algorithm-Hardware Co-Design: CNN Optimization & C++ Memory Tuning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Bazel](https://img.shields.io/badge/Bazel-43A047?style=for-the-badge&logo=bazel&logoColor=white)
![Computer Architecture](https://img.shields.io/badge/Domain-Computer_Architecture-blue?style=for-the-badge)

## 📌 Overview

This project showcases a full-stack approach to **Algorithm-Hardware Co-Design**, bridging high-level Deep Learning architectures with low-level system optimization. By targeting the core bottlenecks of modern AI accelerators—computational complexity and memory bandwidth—this repository demonstrates how hardware-aware software design can drastically improve performance.

### ✨ Key Highlights
* **Mathematical Rigor:** Derived backpropagation gradients from scratch ($dL/dZ = P - Y$) and implemented custom CNN filters without relying on high-level wrappers.
* **Algorithm Optimization:** Reduced Multiply-Accumulate (MAC) operations and model parameters via mathematical filter decomposition (e.g., spatially separable convolutions).
* **Hardware-Level Tuning:** Maximized CPU computation throughput in C++ Matrix Multiplication (GEMM) by exploiting Spatial Locality and L1 Cache Tiling.

---

## 🚀 Technical Deep Dive

### 1. Hardware-Aware Algorithm Optimization
* **Filter Decomposition:** Significantly reduced computational overhead by mathematically breaking down a large $5\times5$ convolutional kernel into two sequential $3\times3$ kernels, and further into **spatially separable convolutions** ($5\times1$ followed by $1\times5$).
* **Easing Memory Bandwidth:** Demonstrated that this decomposition maintains the original Receptive Field while drastically cutting down MAC operations—a critical technique for reducing power consumption and memory bandwidth pressure in IC design and AI accelerators.

### 2. C++ Memory Hierarchy & Cache Tuning
* **Spatial Locality & Prefetching:** Conducted comprehensive benchmarks on C++ GEMM. Proved that the `kij` loop order vastly outperforms the intuitive `ijk` order by perfectly aligning with "Row-Major" memory storage and hardware prefetcher patterns.
* **Cache Tiling (Blocking):** Resolved Cache Thrashing in large matrix operations by implementing Tiling to enhance Temporal Locality. Empirically identified the optimal Tile Size (e.g., $T=64$) to fit working memory blocks perfectly into the L1 Data Cache.

### 3. CNN Architecture & Training Pipeline
* **Low-Level Implementation:** Built classic image processing filters (Average, Gaussian, Laplacian, Edge Detection) from scratch using PyTorch tensor operations (`conv2d`).
* **End-to-End LeNet-5:** Engineered a complete optimization pipeline for CIFAR-10 classification, fully managing the forward pass, Cross-Entropy Loss computation, and backpropagation mechanics.

---

## 📂 Project Structure

```text
├── cpp_files/               # C++ implementations for GEMM and memory tuning
│   ├── files/1/             # C++ source code and headers
│   ├── .bazelrc             # Bazel build configuration settings
│   ├── MODULE.bazel         # Bazel module dependencies
│   ├── WORKSPACE            # Bazel workspace definition
│   ├── image.png            # Benchmark/Performance visualization
│   └── image1.png           # Benchmark/Performance visualization
├── files/                   # Python implementation and experiments
│   └── cnn.ipynb            # Jupyter Notebook: PyTorch CNN & filter decomposition
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
