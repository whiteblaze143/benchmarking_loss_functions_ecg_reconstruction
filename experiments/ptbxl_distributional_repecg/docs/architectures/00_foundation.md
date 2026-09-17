# The repECG Program: Foundational Pipeline

Welcome to the **repECG** (Distributional Representation of the Electrocardiogram) program. If you are a new graduate student, researcher, or engineer stumbling upon this repository with no prior context, this document is designed to teach you everything you need to know. 

## Introduction: The Philosophy of repECG

### The Problem with Standard Deep Learning in Cardiology
In standard deep learning applied to electrocardiograms (ECGs), researchers typically feed raw voltage sequences (e.g., a 1D sequence of length 5000 samples) directly into a massive Convolutional Neural Network (CNN) or Transformer. 

While this produces high accuracy on the specific training dataset (like PTB-XL), it suffers from severe **shortcut learning**. Deep neural networks are lazy; they will take the easiest mathematical path to reduce their error. If all patients with myocardial infarction (heart attacks) in a specific dataset happened to be recorded on an older ECG machine that introduces a specific high-frequency noise, the neural network will just memorize the noise signature of the machine instead of learning what a heart attack looks like. It will also memorize baseline wander (from breathing) or analog filter artifacts.

When you take this model and deploy it in a real hospital, it collapses and fails because it never learned the true electrophysiology of the heart.

### The repECG Solution
**Instead of treating the ECG as a deterministic squiggly line, we treat it as a sequence of statistical distributions.** 

A healthy P-wave (the contraction of the upper chambers of the heart) is not defined by exact, rigid voltage coordinates. It is defined by a probability distribution of electrical vectors. By mapping these empirical distributions into a rigorous mathematical space—specifically, a Reproducing Kernel Hilbert Space (RKHS)—we force the neural networks to learn the true biological mechanics rather than hospital-specific noise.

---

## The Foundational Pipeline: From Raw Voltage to RKHS

Before any of our 15 experimental models touch the data, the raw ECG goes through a strict, mathematically grounded, shared preprocessing pipeline. Understanding this pipeline is critical, as it is the foundation for the entire repository.

### 1.1 Physical Voltage Preservation and the 8-Lead Basis
Most deep learning for computer vision normalizes image pixels to a `[-1, 1]` or `[0, 1]` range. **We explicitly forbid this in repECG.** 
The electrical amplitude (measured in millivolts, mV) of the heart is biologically critical. For example, left ventricular hypertrophy (a thickened heart muscle) directly causes larger electrical voltages because there is more muscle mass firing. If we normalize the signal, we destroy this diagnostic information. We strictly preserve the physical mV scale.

Furthermore, a standard 12-lead ECG is highly redundant. It is simply 12 different camera angles looking at the same 3D electrical dipole vector of the heart. We apply a linear spatial transformation to reduce the 12 leads into a mathematically independent **8-lead orthogonal basis**, perfectly capturing the entire electrical field while removing noise and redundancy.

### 1.2 Phase Alignment and "Phase Cells"
The heart is cyclic and dynamic. A patient resting might have a heart rate of 60 Beats Per Minute (BPM), meaning a beat lasts 1 second. A running patient might have a heart rate of 120 BPM, meaning a beat lasts 0.5 seconds.

Comparing a P-wave at absolute timestamp $t=200ms$ in Patient A to the same absolute timestamp in Patient B is biologically meaningless, because Patient B's heart beats faster, so $t=200ms$ might be the T-wave for them.

**The Solution:**
1. **R-Peak Detection**: We detect the R-peaks (the massive voltage spikes representing the main ventricular contraction) to reliably segment the continuous signal into valid heartbeat cycles (RR-intervals).
2. **Phase Resampling**: We uniformly resample every single valid heartbeat into a fixed mathematical grid of **256 phase samples**. Now, phase index 10 is *always* the beginning of the beat, and phase index 250 is *always* the end of the beat, completely invariant to the patient's heart rate.
3. **Phase Cells**: We chop this 256-sample standardized cycle into short, locally contiguous chunks called **Phase Cells** (e.g., 16 samples wide). These phase cells are our fundamental units of analysis. Think of a phase cell as a short, localized burst of cardiac activity representing a specific physiological event.

### 1.3 Kernel Mean Embeddings (KME) and the Nyström Approximation
This is the most mathematically crucial step of the entire repECG pipeline. We **do not** feed the raw voltages of the phase cells into the neural networks. Instead, we represent each phase cell as a **probability distribution** of voltages.

#### The Math: RKHS
- **The Problem:** How do you feed a "probability distribution" into a neural network?
- **The Solution (KME):** We map the empirical distribution of the phase cell into a Reproducing Kernel Hilbert Space (RKHS). In this mathematically infinite-dimensional space, an entire probability distribution becomes a single, deterministic point, called the Kernel Mean Embedding (KME). If two phase cells have similar underlying distributions of voltages, their KME points will be close together in the RKHS.

#### The Computational Reality: Nyström Method
- Because computers cannot process infinite-dimensional vectors, we must approximate the RKHS. We use the **Nyström method**.
- During training, we extract millions of phase cells and run K-means clustering to find $K$ (e.g., 256) representative "anchor" distributions.
- For any new phase cell, we compute its kernel similarity to these 256 anchors. This gives us a stable, finite-dimensional vector (e.g., a 256-dimensional vector) that perfectly and rigorously describes the distribution of that phase cell. 

**Result:** Every downstream model in the repECG program takes as input a sequence of these stable, finite-dimensional KME vectors, entirely bypassing raw voltage noise.
