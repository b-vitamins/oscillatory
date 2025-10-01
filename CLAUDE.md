# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oscillatory Neural Networks in PyTorch - implements Adaptive Kuramoto Oscillatory Recurrent Networks (AKOrN) and Kuramoto Orientation Diffusion Models (KODM) for computer vision tasks.

## Core Architecture

### Package Structure
- `oscillatory.nn.functional`: Mathematical operations for oscillatory dynamics
  - `kuramoto`: Kuramoto oscillator dynamics, normalization, tangent space projections
  - `diffusion`: Phase diffusion operations for circular variables (S¹)
  - `gta`: Group-theoretic attention with SO(2) rotation matrices
  - `utils`: Positional encoding utilities
- `oscillatory.nn.modules`: Neural network modules
  - `akorn`: Three variants - Hierarchical (multi-scale vision), Dense (classification), Grid (spatial reasoning)
  - `kodm`: Diffusion models for circular orientation data with UNet architecture
  - `attention`: GTAttention implementation
  - `heads`: Task-specific heads (classification, object discovery, Sudoku)

### Key Concepts

**Kuramoto Oscillators**: Groups of n_oscillators per channel that evolve on unit hypersphere. Each group represents complex-valued features through n-dimensional unit vectors. Dynamics combine:
- Natural frequency (omega): Rotational term causing oscillation
- Coupling: Connectivity-driven synchronization (conv or attention)
- Stimulus: External input signal
- Tangent space projection: Ensures updates remain on sphere

**AKOrN Variants**:
- `AKOrNHierarchical`: Multi-scale hierarchical processing with spatial downsampling stages
- `AKOrNDense`: Flat architecture with patchification for image classification
- `AKOrNGrid`: Grid-structured for spatial reasoning (Sudoku, etc.)

**KODM**: Treats image pixels as phase angles θ ∈ [-π, π], combines wrapped Gaussian noise with Kuramoto mean-field coupling in diffusion process. Forward diffusion adds noise gradually; reverse diffusion uses UNet to predict score function.

**Readout**: Extracts scalar features from oscillator states via L2 norm: ||oscillators||₂

## Development Commands

### Environment Setup
All commands must run through Guix shell:
```bash
guix shell -m manifest.scm -- <command>
```

### Testing
```bash
# Run all tests with coverage
guix shell -m manifest.scm -- pytest -xvs --cov=oscillatory --cov-report=term-missing

# Run specific test categories (using markers from pytest.ini)
guix shell -m manifest.scm -- pytest -xvs -m unit
guix shell -m manifest.scm -- pytest -xvs -m integration
guix shell -m manifest.scm -- pytest -xvs -m property
guix shell -m manifest.scm -- pytest -xvs -m performance

# Run single test
guix shell -m manifest.scm -- pytest -xvs tests/unit/functional/test_kuramoto.py::test_normalize_oscillators2d

# Skip slow tests
guix shell -m manifest.scm -- pytest -xvs -m "not slow"
```

### Code Quality
```bash
# Format and lint (run before commits)
guix shell -m manifest.scm -- ruff format .
guix shell -m manifest.scm -- ruff check .
guix shell -m manifest.scm -- pytest -xvs
```

### Test Organization
- `tests/unit/`: Component-level tests for individual functions and modules
- `tests/integration/`: End-to-end pipeline tests
- `tests/property/`: Mathematical invariant tests (e.g., sphere normalization)
- `tests/performance/`: Benchmark and performance regression tests
- `tests/conftest.py`: Shared fixtures including device, dtype, gradient checking utilities

### Key Implementation Details

**Channel Grouping**: Channels C must be divisible by n_oscillators. Internal reshaping: [B, C, H, W] → [B, C//n_osc, n_osc, H, W]

**Omega (Natural Frequency)**:
- Applies SO(2) rotation to adjacent oscillator pairs
- Shape: scalar (global_omega=True) or [n_groups] (global_omega=False)
- Only supported for even n_oscillators

**Connectivity Types**:
- 'conv': Spatial convolution with kernel_size
- 'attn': Multi-head attention with optional Group-Theoretic Attention (use_gta=True)

**Random Initialization**: Models initialize oscillator state x with `torch.manual_seed(999999)` followed by `torch.randn_like(c)` for reproducibility

**Integration Step**: Uses explicit Euler with configurable gamma parameter. Update: x_{t+1} = normalize(x_t + gamma * (omega_term + proj(coupling + stimulus)))

## Dependencies

PyTorch 2.7.0 with CUDA support, numpy, scipy, scikit-learn, matplotlib, pandas, einops. All managed via manifest.scm - never use pip directly.
