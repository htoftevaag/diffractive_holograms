# diffractive_holograms
diffractive_hologram is a hobby project exploring the use of the Gerchberg–Saxton (G-S) algorithm to design diffractive optical elements (DOEs) for generating holograms. The project focuses on both transmissive and reflective holographic elements and provides a simple workflow for simulating, iterating, and visualising phase-only holograms.

![Computer-generated hologram through diffractive optical elements using the G-S algorithm](https://github.com/htoftevaag/diffractive_holograms/blob/main/figures/DOE_fig.png?raw=true)

## ✨ Features

- Gerchberg–Saxton algorithm implementation
- Generates phase-only holograms from target intensity patterns
- Support for transmissive and reflective DOEs
- Includes simulation paths for both types of optical configurations
- Fourier-based optical propagation
- Uses FFT routines to model far‑field diffraction and hologram reconstruction.
- Visualization tools
- Plots amplitude, phase, reconstruction results, and error metrics.
- Modular code structure
- Easy to modify for different wavelengths, sampling parameters, and target images.

## 🔬 What the project is about

The goal of this project is to experiment with computational holography and diffractive optics using accessible numerical methods. While not meant for production‑grade optical design, it provides a flexible playground for studying how the G-S algorithm behaves under different constraints and for creating custom holographic patterns.

## ⚙️ Typical workflow

- Load or generate a target intensity pattern for the hologram
- Run the G-S iterative phase retrieval loop
- Output a phase-only DOE suitable for:
  - Transmissive holograms (e.g., etched or printed phase masks)
  - Reflective holograms (e.g., reflective SLMs or metallic DOEs)
- Visualize reconstructed holograms and error convergence.

## 📁 Contents

- gs_algorithm.py – core G‑S implementation
- simulation.py – propagation and reconstruction utilities
- examples/ – demonstration notebooks and example targets
- figures/ – plots of phase masks and reconstructions

## 🛠️ Future ideas

- Multi-wavelength or color holograms
- GPU acceleration

## Other alternatives
Other open-source packages include:
- pyoptools https://github.com/cihologramas/pyoptools 
- HoloGen https://gitlab.com/CMMPEOpenAccess/HoloGen 
- Computer-Generated-Hologram https://github.com/JackHCC/Computer-Generated-Hologram
