# FMClosure

Learning PDE time stepping with flow matching.

This repository provides tools for learning partial differential equation (PDE) time-stepping using flow matching techniques. The implementation is based on a UNet architecture and flow-matching principles, inspired by the course [IAP Diffusion Labs](https://github.com/eje24/iap-diffusion-labs). For a Julia adaptation, see [FlowMatching](https://github.com/agdestein/FlowMatching).

## Features
- Implements flow matching for PDE time-stepping.
- Utilizes UNet architecture for learning.
- Includes examples for Burgers' equation, Kuramoto-Sivashinsky (KS) equation, and Korteweg-de Vries (KdV) equation.
- Pre-trained parameter files for various configurations.

## Prerequisites
- Julia 1.12.5 or later.
- Basic knowledge of Julia and PDEs.
- Familiarity with flow matching and UNet architectures is helpful but not required.

## Installation
To set up the project, clone the repository and instantiate the Julia environment:

```bash
# Clone the repository
git clone https://github.com/your-repo/FMClosure.git
cd FMClosure

# Instantiate the Julia environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running the Code
The repository includes several main scripts for running experiments:

- `main_KS.jl`: Run simulations for the Kuramoto-Sivashinsky equation.
- `main_burgers.jl`: Run simulations for Burgers' equation.
- `main.jl`: General entry point for other experiments.

To execute a script, use the Julia REPL:

```bash
julia --project=. main_KS.jl
```

## Data Files
The repository includes several `.jld2` files for pre-trained parameters and datasets:
- `myparameters_*.jld2`: Pre-trained parameters for various equations and configurations.
- `mydata_*.jld2`: Training and testing datasets.

## References
- [IAP Diffusion Labs](https://github.com/eje24/iap-diffusion-labs)
- [FlowMatching](https://github.com/agdestein/FlowMatching)

## License
This project is licensed under the MIT License. See the LICENSE file for details.