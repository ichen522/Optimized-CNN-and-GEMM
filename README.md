## 💡 What I Learned

Through this project, I gained a deep understanding of the complete **algorithm-hardware co-design** workflow, bridging the gap from high-level neural network architecture design to low-level C++ memory access optimization:

### 1. CNN & Computer Vision Foundations
* **Low-Level Filter Implementation**: Rather than relying on high-level wrappers, I used PyTorch tensor operations and `conv2d` to implement classic image processing filters from scratch, including Average, Gaussian, Laplacian, and Edge Detection filters.
* **Model Construction & Training Loop**: Built a LeNet-5 style Convolutional Neural Network for CIFAR-10 image classification. I fully implemented the optimization pipeline, including the forward pass, Cross-Entropy Loss computation, backpropagation, and evaluation inference using `torch.no_grad()`.
* **Math & Gradient Derivation**: Deepened my understanding of the mathematical advantages of combining Softmax with Cross-Entropy Loss, and successfully derived its elegantly simple gradient formula ($dL/dZ = P - Y$).

### 2. Algorithm-Level Hardware Optimization (Filter Decomposition)
* **Reducing Computational Complexity**: Implemented **filter decomposition** techniques, breaking down a large $5\times5$ convolutional kernel into two $3\times3$ kernels, and even further into **spatially separable convolutions** ($5\times1$ and $1\times5$).
* **Hardware Awareness**: Understood that this algorithm-level decomposition not only maintains the same Receptive Field but also drastically reduces the number of model parameters and Multiply-Accumulate (MAC) operations. This is crucial for easing memory bandwidth pressure and reducing power consumption in AI accelerator chips.

### 3. C++ Matrix Operations & Memory Hierarchy Tuning
* **Spatial Locality**: In the C++ Matrix Multiplication (GEMM) benchmarks, I experienced firsthand the massive performance impact of "Row-Major" memory storage in C++. I verified that the `kij` loop order vastly outperforms the intuitive `ijk` order because it aligns perfectly with the hardware prefetcher's contiguous memory access patterns.
* **Cache Tiling / Blocking**: To solve the Cache Thrashing problem caused by large matrix operations, I implemented Tiling to improve Temporal Locality. Through experimentation, I identified the optimal Tile Size (e.g., T=64) that perfectly fits the working blocks into the L1 Data Cache, maximizing CPU computation throughput.

---
