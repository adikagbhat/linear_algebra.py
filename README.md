# Numerical Computing & Linear Algebra Utilities

A lightweight Python library providing deterministic implementations of fundamental numerical methods, matrix factorizations, and error verification benchmarks.

## Features
- **LU Matrix Factorization:** Doolittle's algorithm decomposing a square matrix into Lower and Upper triangular components ($A = LU$).
- **Linear Solver:** Forward and backward substitution for direct solution of linear systems ($Ax = b$).
- **Deterministic Convergence:** Automated $L_2$ norm residual checking ($\|Ax - b\|_2 < \epsilon$) ensuring numerical stability.

## Tech Stack
- **Language:** Python 3.8+
- **Core Library:** NumPy

## Quickstart
```bash
git clone https://github.com/adikagbhat/linear_algebra.py.git
cd linear_algebra.py
```
