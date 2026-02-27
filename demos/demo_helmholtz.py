import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import Function, functionspace, form, create_interpolation_data
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_submesh
from ufl import dx, grad, inner, Measure
from mpi4py import MPI
from petsc4py import PETSc
from scifem import PointSource
from utils import MicrophonePressure, plot_complex_spectra
from fenicsx_pml import LcPML

#frequency range definition
frequencies = np.arange(100, 5000, 200)

# fluid quantities definition
c0 = 340
rho_0 = 1.225

# PML generation
pml = LcPML(
    filename="../test/box.msh",
    d_pml=0.03, 
    n_layers=3 # depending on physics, increase both d_pml and n_layers
) 
msh, cell_tags, facet_tags = pml.generate(physical_group=3)
pml.compute_pml_properties()

# Source amplitude and position
Q = 1e-4

if msh.comm.rank == 0:
    x_S   = np.array([0.00, 0.0, 0.09]) 
else:
    x_S = np.zeros((0, 3))

# Test and trial function spaces
deg = 2
V = functionspace(msh, ("CG", deg))
p = ufl.TrialFunction(V)
v = ufl.TestFunction(V)  

# PoinSource (Monopole)
b = Function(V)
point_source = PointSource(V, x_S, magnitude=1)
point_source.apply_to_vector(b)
b.x.scatter_reverse(la.InsertMode.add)
b.x.scatter_forward()

# Weak Form
dx = Measure("dx", domain=msh, subdomain_data=cell_tags, metadata={"quadrature_degree": 3*deg})

a  = (
    inner(grad(p), grad(v)) * dx(1) - pml.k0**2 * inner(p, v) * dx(1) # Air domain
  + inner(pml.Lambda_PML*grad(p), grad(v)) * dx(2) - pml.detJ * pml.k0**2 * inner(p, v) * dx(2) # PML domain 
)

a_form = form(a) 
A = assemble_matrix(a_form, bcs=[])
A.assemble()

# S
ph      = Function(V)
ph.name = "p"

# Solver setup
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")

# Creation of the submesh without the PML on which the soultion is projected
i_INT = cell_tags.indices[(cell_tags.values==1)] # 
msh_INT, entity_map, vertex_map, geom_map = create_submesh(msh, msh.topology.dim, i_INT)
V_INT = functionspace(msh_INT, ("CG", deg))
ph_nopml = Function(V_INT)

fine_mesh_cell_map = msh_INT.topology.index_map(msh_INT.topology.dim)
num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
interp_cells = np.arange(num_cells_on_proc, dtype=np.int32)
interpolation_data = create_interpolation_data(V_INT, V, interp_cells, padding=1e-14)

# Spectrum initialization
p_mic = np.zeros((len(frequencies),1),dtype=complex)

# Microphone 
x_mic = np.array([0.0, 0.0, 0.0]) 

mic = MicrophonePressure(msh, x_mic)

# Export setup (for the PML removed for nice visual)
export_fields = True
if export_fields:
    vtx_nopml = VTXWriter(msh_INT.comm, "fields/p_nopml.bp", [ph_nopml], engine="BP4")


for nf, f in enumerate(frequencies):
    if msh.comm.rank == 0:
        print(f"Computing Frequency: {f}", flush=True)
    
    omega = 2 * np.pi * f

    # PML forms are frequency dependent. Update k0: IMPORTANT!!!
    pml.k0.value = omega / c0

    # Reassemble matrix
    A.zeroEntries()
    assemble_matrix(A, a_form)
    A.assemble()
    
    # Define load vector
    jrwQb = 1j * rho_0 * omega * Q * b.x.petsc_vec

    # Solve
    ksp.solve(jrwQb, ph.x.petsc_vec)
    ph.x.scatter_forward()
    
    # Interpolate solution on subdomain (without PML)
    ph_nopml.interpolate_nonmatching(ph,
                                     interp_cells,
                                     interpolation_data)
    
    # Export fields (optional)
    if export_fields:
        vtx_nopml.write(f)
    
    # Evaluate pressure at mic location
    p_f = mic.listen(ph)
    p_f = msh.comm.gather(p_f, root=0)
    
    # Fill spectrum
    if msh.comm.rank == 0: 
        assert p_f is not None
        p_mic[nf] = np.hstack(p_f)


# Plot computed spectrum against Monopole analytic formula
import matplotlib.pyplot as plt

def monopole_pressure(f, r, Q, c=340, rho=1.225):
    omega = 2 * np.pi * f
    k = omega / c

    return (1j * omega * rho * Q / (4 * np.pi * r)) * np.exp(-1j * k * r)

if msh.comm.rank == 0:
    R = np.linalg.norm(x_mic - x_S)
    p_analytic = monopole_pressure(frequencies, R, 1e-4)
    
    # Custom tool to plot amplitude and phase
    plot_complex_spectra(
        x_axis=frequencies,
        p_spectra_list=[p_mic, p_analytic],
        labels=["Computed", "Analytical"],
        plot_db=True
    )