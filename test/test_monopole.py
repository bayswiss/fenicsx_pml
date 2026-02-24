import numpy as np
import ufl
import pytest
from dolfinx.fem import Function, functionspace, assemble_scalar, form, create_interpolation_data
from dolfinx.fem.petsc import assemble_matrix, LinearProblem
from dolfinx.mesh import create_submesh
from ufl import dx, grad, inner, Measure
from mpi4py import MPI
from petsc4py import PETSc
from scifem import PointSource
from fenicsx_pml import LcPML

def test_monopole_accuracy():
    comm = MPI.COMM_WORLD
    f = 1000.0  # Test at a single representative frequency
    c0, rho_0 = 340.0, 1.225
    omega = 2 * np.pi * f
    k0_val = omega / c0
    Q = 1e-4

    x_S_exact = np.array([0.00, 0.0, 0.09])
    
    # 2. Define the rank-dependent array strictly for the FEM assembly
    if comm.rank == 0:
        x_S_source = x_S_exact
    else:
        x_S_source = np.zeros((0, 3))
    
    # 1. Setup PML and Mesh
    pml = LcPML("air.msh", 0.03, 3, comm=comm)
    msh, cell_tags, facet_tags = pml.generate(physical_group=3)
    pml.compute_pml_properties()
    pml.k0.value = k0_val

    # 2. Setup Function Spaces and Forms
    deg = 2
    V = functionspace(msh, ("Lagrange", deg))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)  

    # Source vector
    b = Function(V)
    point_source = PointSource(V, x_S_source, magnitude=1)
    point_source.apply_to_vector(b)

    dx_m = Measure("dx", domain=msh, subdomain_data=cell_tags, metadata={"quadrature_degree": 3*deg})

    a = (inner(grad(u), grad(v)) - pml.k0**2 * inner(u, v)) * dx_m(1) + \
        (inner(pml.Lambda_PML * grad(u), grad(v)) - pml.detJ * pml.k0**2 * inner(u, v)) * dx_m(2)

    # 3. Assemble and Solve
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

    # 4. Extract Interior Domain
    i_INT = cell_tags.indices[(cell_tags.values == 1)]
    msh_INT, entity_map, vertex_map, geom_map = create_submesh(msh, msh.topology.dim, i_INT)
    V_INT = functionspace(msh_INT, ("Lagrange", deg))
    
    uh_INT = Function(V_INT)
    interp_cells = np.arange(V_INT.mesh.topology.index_map(3).size_local, dtype=np.int32)
    interp_data = create_interpolation_data(V_INT, V, interp_cells, padding=1e-14)
    uh_INT.interpolate_nonmatching(uh, interp_cells, interp_data)

    # 5. Compute Exact Analytical Solution on Interior
    u_exact = Function(V_INT)
    def exact_solution(x):
        # Use the global x_S_exact here!
        r = np.linalg.norm(x - x_S_exact[:, None], axis=0) 
        r = np.where(r < 1e-12, 1e-12, r)
        return 1j * rho_0 * omega * Q * np.exp(1j * k0_val * r) / (4 * np.pi * r)
    
    u_exact.interpolate(exact_solution)
    u_exact.x.scatter_forward()
    
    from dolfinx.io import VTXWriter

    error_func = Function(V_INT)
    error_func.x.array[:] = u_exact.x.array - uh_INT.x.array
    error_func.x.scatter_forward()
    error_func.name = "error"
    vtx = VTXWriter(msh_INT.comm, "fields/test.bp", [uh_INT, u_exact], engine="BP4")

    vtx.write(0)

    # 6. Evaluate Errors with an Exclusion Zone
    dx_int = Measure("dx", domain=msh_INT)
    
    # Define an exclusion radius (e.g., 2 or 3 times your mesh size 'h')
    r_excl = 0.02 
    
    # Create UFL mask for L2 integration
    x_ufl = ufl.SpatialCoordinate(msh_INT)
    r_ufl = ufl.sqrt((x_ufl[0] - x_S_exact[0])**2 + (x_ufl[1] - x_S_exact[1])**2 + (x_ufl[2] - x_S_exact[2])**2)
    mask = ufl.conditional(ufl.ge(ufl.real(r_ufl), r_excl), 1.0, 0.0)

    # Compute masked Absolute L2 Error
    error = uh_INT - u_exact
    l2_err_sq = assemble_scalar(form(inner(error, error) * mask * dx_int))
    l2_err_sq = comm.allreduce(l2_err_sq, op=MPI.SUM)
    l2_err = np.sqrt(np.real(l2_err_sq))
    
    # Compute masked Exact L2 Norm (for relative error)
    l2_exact_sq = assemble_scalar(form(inner(u_exact, u_exact) * mask * dx_int))
    l2_exact_sq = comm.allreduce(l2_exact_sq, op=MPI.SUM)
    l2_exact = np.sqrt(np.real(l2_exact_sq))
    
    rel_l2_err = l2_err / l2_exact if l2_exact > 0 else 0.0
    
    # Compute masked Max Error using NumPy
    coords = V_INT.tabulate_dof_coordinates()
    dists = np.linalg.norm(coords - x_S_exact, axis=1)
    valid_dofs = dists >= r_excl
    
    if np.any(valid_dofs):
        local_max = np.max(np.abs(uh_INT.x.array[valid_dofs] - u_exact.x.array[valid_dofs]))
    else:
        local_max = 0.0
        
    max_err = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        print(f"\nFrequency: {f} Hz")
        print(f"L2 Error:       {l2_err:.4e}")
        print(f"Relative L2:    {rel_l2_err:.4%}")
        print(f"Max Abs Error:  {max_err:.4e}")

    # 7. Assertions (Set your acceptable tolerances here based on mesh size h)
    assert rel_l2_err < 0.05, f"Relative L2 error too high: {rel_l2_err:.2%}"
    # max_err is highly dependent on how close the source is to a node, so use a generous bound
    assert max_err < 1.0, f"Max error exploded: {max_err:.2e}"