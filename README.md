# FEniCSx_PML
A FEniCSx tool to generate locally conformal PML layers from Gmsh meshes for use in frequency domain wave simulations.

Compatible with dolfinx **v0.9.0** and **v0.10.0**

<img width="655" height="800" alt="fenicsx_pml2" src="https://github.com/user-attachments/assets/7382c67b-ca6f-483a-bfd4-101fe24948b5" />

## Prerequisites

This tool requires **FEniCSx** with complex build of PETSc. We recommend installing it via ```conda```.

**Install FEniCSx via Conda:**
    
    ```bash
    conda create -n fenicsx-env
    conda activate fenicsx-env
    conda install -c conda-forge fenics-dolfinx mpich petsc=*=complex* 
    ```
Alternative ways to install FEniCSx: [https://github.com/FEniCS/dolfinx?tab=readme-ov-file#installation](https://github.com/FEniCS/dolfinx?tab=readme-ov-file#installation)
## Installation

Once you are inside your FEniCSx environment, download this repository and install it:

**Clone the repository:**
  
```bash
git clone https://github.com/bayswiss/fenicsx_pml.git
cd fenicsx_pml
```

**Install using pip:**
```bash
pip install .
```

**For developers (allows you to edit code without reinstalling):**
```bash
pip install -e .
```

## Usage

```python

from fenicsx_pml import LcPML

# Import mesh, generate layers
LcPML(filename="name.msh", d_pml=0.03, n_layers = 3)
msh, cell_tags, facet_tags = pml.generate(physical_group=3)
pml.compute_pml_properties()

# PML functions to use into the weak form: 
Lambda_PML = pml.Lambda_PML 
detJ = pml.detJ

# ... weak form, assemble and solve ...
```
 
---

### References
<small>
The LC PML and underlying coordinate stretching implemented in this module are based on the theory and methodologies presented in the following works (along with other related publications by the same authors):

* **[1]** Y. Mi, X. Yu. *Isogeometric locally-conformal perfectly matched layer for time-harmonic acoustics*, 2021.
* **[2]** H. Bériot, A. Modave. *An automatic perfectly matched layer for acoustic finite element simulations in convex domains of general shape*, 2020.
* **[3]** O. Ozgun, M. Kuzoglu. *Locally-Conformal Perfectly Matched Layer implementation for finite element mesh truncation*, 2006.
</small>
