# SD-GLE: Spatial Disorder Generalized Langevin Equation
## Overview
This repository implements the SD-GLE framework, a machine learning approach for *Coarse-Grained Dynamics with Spatial Disorder and Non-Markovian Memory*.

Complex environments, such as glass-forming liquids or disordered media, present two major challenges for coarse-grained modeling: rugged local potential energy surfaces (static spatial disorder) and non-Markovian memory effects (viscoelasticity). This framework simultaneously learns both components directly from particle trajectory data by coupling a non-Markovian Generalized Langevin Equation (GLE) with a Gaussian random field.

## Installation
```bash
# Clone the repository
git clone https://github.com/aweqardf/Spatial_disorder_GLE.git
cd Spatial_disorder_GLE

# Install required dependencies
pip install numpy scipy torch gpytorch
```
