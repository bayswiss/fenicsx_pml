import numpy as np
import ufl
import pytest
from dolfinx.fem import Function, functionspace, assemble_scalar, form, create_interpolation_data
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import create_submesh
from dolfinx.io import VTXWriter
from ufl import dx, grad, inner, Measure
from mpi4py import MPI
from petsc4py import PETSc
from scifem import PointSource
from fenicsx_pml import LcPML

def test_monopole_accuracy():
    comm = MPI.COMM_WORLD
    f = 1000.0  
    c0, rho_0 = 340.0, 1.225
    omega = 2 * np.pi * f
    k0_val = omega / c0
    Q = 1e-4

    # The true coordinates for exact math
    x_S_exact = np.array([0.00, 0.0, 0.09])
    
    # Your original rank-dependent source logic
    if comm.rank == 0:
        x_S_source = x_S_exact
    else:
        x_S_source = np.zeros((0, 3))
    
    pml = LcPML("air.msh", 0.03, 3, comm=comm)
    msh, cell_tags, facet_tags = pml.generate(physical_group=3)
    pml.compute_pml_properties()
    pml.k0.value = k0_val

    deg = 2
    V = functionspace(msh, ("Lagrange", deg))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)  

    b = Function(V)
    point_source = PointSource(V, x_S_source, magnitude=1)
    point_source.apply_to_vector(b)

    dx_m = Measure("dx", domain=msh, subdomain_data=cell_tags, metadata={"quadrature_degree": 3*deg})

    a = (inner(grad(u), grad(v)) - pml.k0**2 * inner(u, v)) * dx_m(1) + \
        (inner(pml.Lambda_PML * grad(u), grad(v)) - pml.detJ * pml.k0**2 * inner(u, v)) * dx_m(2)

    a_form = form(a)
    A = assemble_matrix(a_form, bcs=[])
    A.assemble()

    uh = Function(V)
    jrwQb = 1j * rho_0 * omega * Q * b.x.petsc_vec
    
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.solve(jrwQb, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # --- YOUR ORIGINAL SUBMESH LOGIC ---
    i_INT = cell_tags.indices[(cell_tags.values == 1)]
    msh_INT, entity_map, vertex_map, geom_map = create_submesh(msh, msh.topology.dim, i_INT)
    V_INT = functionspace(msh_INT, ("Lagrange", deg))
    
    uh_INT = Function(V_INT)
    interp_cells = np.arange(V_INT.mesh.topology.index_map(3).size_local + V_INT.mesh.topology.index_map(3).num_ghosts, dtype=np.int32)
    interp_data = create_interpolation_data(V_INT, V, interp_cells, padding=1e-14)
    uh_INT.interpolate_nonmatching(uh, interp_cells, interp_data)
    uh_INT.x.scatter_forward()

    # Exact solution on the submesh
    u_exact = Function(V_INT)
    def exact_solution(x):
        r = np.linalg.norm(x - x_S_exact[:, None], axis=0) 
        r = np.where(r < 1e-12, 1e-12, r)
        return 1j * rho_0 * omega * Q * np.exp(1j * k0_val * r) / (4 * np.pi * r)
    
    u_exact.interpolate(exact_solution)
    u_exact.x.scatter_forward()
    
    '''
    E' UN PUTTANAIO. USALO COME SCHELETRO, MA SEGUI IL CODICE SCRITTO DA TE, QUA NON FUNZIONA UN CAZZO
    '''

    # Export exactly as you had it
    vtx_nopml = VTXWriter(msh_INT.comm, "fields/u_nopml.bp", [uh_INT, u_exact], engine="BP4")
    vtx_nopml.write(0)
    vtx_nopml.close()

    # --- NUMPY ERROR CALCULATION (No UFL Masks needed) ---
    r_excl = 0.02
    coords = V_INT.tabulate_dof_coordinates()
    dists = np.linalg.norm(coords - x_S_exact, axis=1)
    
    # Mask out the singularity purely in NumPy
    valid_dofs = dists >= r_excl
    
    # Max Error
    if np.any(valid_dofs):
        local_max = np.max(np.abs(uh_INT.x.array[valid_dofs] - u_exact.x.array[valid_dofs]))
    else:
        local_max = 0.0
    max_err = comm.allreduce(local_max, op=MPI.MAX)
    
    # Relative L2 Error (Discrete approximation over valid DOFs)
    if np.any(valid_dofs):
        local_l2_num = np.sum(np.abs(uh_INT.x.array[valid_dofs] - u_exact.x.array[valid_dofs])**2)
        local_l2_den = np.sum(np.abs(u_exact.x.array[valid_dofs])**2)
    else:
        local_l2_num, local_l2_den = 0.0, 0.0
        
    global_l2_num = comm.allreduce(local_l2_num, op=MPI.SUM)
    global_l2_den = comm.allreduce(local_l2_den, op=MPI.SUM)
    
    rel_l2_err = np.sqrt(global_l2_num) / np.sqrt(global_l2_den) if global_l2_den > 0 else 0.0

    if comm.rank == 0:
        print(f"\nMax Abs Error:  {max_err:.4e}")
        print(f"Relative L2:    {rel_l2_err:.4%}")

    assert rel_l2_err < 0.05, f"Relative L2 error too high: {rel_l2_err:.2%}"
    assert max_err < 1.0, f"Max error exploded: {max_err:.2e}"