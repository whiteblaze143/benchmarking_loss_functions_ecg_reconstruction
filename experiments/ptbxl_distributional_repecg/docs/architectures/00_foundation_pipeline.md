# The repECG Program: Foundational Pipeline

Welcome to the **repECG** (Distributional Representation of the Electrocardiogram) program. 

## Introduction: The Philosophy of repECG
In standard deep learning for electrocardiograms (ECGs), researchers typically feed raw voltage sequences directly into a massive CNN. While this produces high accuracy on the training dataset, it suffers from severe **shortcut learning**. The network memorizes high-frequency noise or analog filter artifacts rather than learning the true electrophysiology of the heart.

Instead of treating the ECG as a deterministic squiggly line, we treat it as a sequence of **statistical distributions**. A healthy P-wave is defined by a probability distribution of electrical vectors. By mapping these distributions into a rigorous mathematical space (a Reproducing Kernel Hilbert Space, or RKHS), we force the neural networks to learn true biological mechanics.

## The Foundational Pipeline (From Voltage to RKHS)
Before any of the 15 models touch the data, the raw ECG goes through a strict, shared preprocessing pipeline. 

### 1. Physical Voltage Preservation and the 8-Lead Basis
Most deep learning normalizes data to a `[-1, 1]` range. **We do not do this.** The electrical amplitude (millivolts, mV) of the heart is biologically critical. We strictly preserve the physical mV scale.
We apply a linear transformation to reduce the redundant 12 leads into a mathematically independent **8-lead orthogonal basis**.

### 2. Phase Alignment and "Phase Cells"
The heart is cyclic. 
1. We detect the R-peaks to segment the signal into valid RR-cycles.
2. We uniformly resample every single heartbeat into a fixed grid of **256 phase samples**.
3. We chop this 256-sample cycle into locally contiguous chunks called **Phase Cells**. These cells are our fundamental units of analysis.

### 3. Kernel Mean Embeddings (KME) and the Nyström Approximation
We represent each phase cell as a **probability distribution** of voltages. We map the distribution into a Reproducing Kernel Hilbert Space (RKHS). In this infinite-dimensional space, the entire probability distribution becomes a single point, called the Kernel Mean Embedding (KME). 
We use the Nyström method to find $K$ anchor points (using K-means clustering) and compute the similarity of our phase cell to these anchors. This gives us a finite, stable vector (e.g., 128 dimensions) that perfectly describes the distribution of the phase cell. 

**Result:** Every paper architecture takes as input a sequence of these stable, finite-dimensional KME vectors.
