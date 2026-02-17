# LcPML Generator

A tool to generate locally conformal PML layers from Gmsh meshes for use in FEniCSx acoustics simulations.

## Prerequisites

This tool requires **FEniCSx**. Because FEniCSx has complex system dependencies (MPI, PETSc), we strictly recommend installing it via Conda first.

1.  **Install FEniCSx via Conda:**
    ```bash
    conda create -n fenicsx-env
    conda activate fenicsx-env
    conda install -c conda-forge fenics-dolfinx mpich petsc=*=complex* 
    ```

## Installation

Once you are inside your FEniCSx environment, download this repository and install it:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/bayswiss/fenicsx_pml.git](https://github.com/bayswiss/fenicsx_pml.git)
    cd fenicsx_pml
    ```

2.  **Install using pip:**
    ```bash
    pip install .
    ```

    *For developers (allows you to edit code without reinstalling):*
    ```bash
    pip install -e .
    ```

## Usage

```python

MWE HERE
```