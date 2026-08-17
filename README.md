# Manifold Learning with Laplacian Eigenmaps

A from-scratch implementation of **Laplacian Eigenmaps** for nonlinear dimensionality reduction and manifold learning.

## What This Project Does

* Implements **k-NN graph construction and graph Laplacian**
* Solves the **generalized eigenvalue problem** for low-dimensional embeddings
* Uses **TWO-NN** to estimate intrinsic dimensionality
* Experiments on a **2D oscillator** and **Yale Face Dataset**

## Results

| Dataset       | Original Dimension | Estimated Intrinsic Dimension |
| ------------- | -----------------: | ----------------------------: |
| 2D Oscillator |                  4 |                         ~2.07 |
| Yale Faces    |             32,256 |                         ~7.01 |

## Tech Stack

**Python · NumPy · SciPy · Scikit-learn · Matplotlib**

## Reference

Laplacian Eigenmaps: Belkin & Niyogi, *Laplacian Eigenmaps for Dimensionality reduction and Data Representation*.
