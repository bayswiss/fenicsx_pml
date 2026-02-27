# Copyright (C) 2026 Antonio Baiano Svizzero
#
# This file is part of FEniCSx_PML (https://github.com/bayswiss/fenicsx_pml)
#
# SPDX-License-Identifier:    GPL-3.0-or-later

import numpy as np
import ufl
from dolfinx import la
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
    """
    Test PML absorption accuracy against an exact analytical acoustic monopole.

    """
    # Init physics
    comm = MPI.COMM_WORLD
    f = 2000.0  
    c0, rho_0 = 340.0, 1.225
    omega = 2 * np.pi * f
    k0_val = omega / c0
    Q = 1e-4

    # Source coords
    x_S_exact = np.array([0.00, 0.0, 0.09])
    
    # Rank 0 source
    if comm.rank == 0:
        x_S_source = x_S_exact
    else:
        x_S_source = np.zeros((0, 3))
    
    # Setup PML
    pml = LcPML("box.msh", 0.03, 3, comm=comm)
    msh, cell_tags, facet_tags = pml.generate(physical_group=3)
    pml.compute_pml_properties()
    pml.k0.value = k0_val

    # FEM space
    deg = 2
    V = functionspace(msh, ("Lagrange", deg))
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)  

    # Point source RHS
    b = Function(V)
    point_source = PointSource(V, x_S_source, magnitude=1)
    point_source.apply_to_vector(b)
    b.x.scatter_reverse(la.InsertMode.add)
    b.x.scatter_forward()
    
    # Integration measure
    dx = Measure("dx", domain=msh, subdomain_data=cell_tags, metadata={"quadrature_degree": 3*deg})

    # Variational form
    a = (inner(grad(p), grad(v)) - pml.k0**2 * inner(p, v)) * dx(1) + \
        (inner(pml.Lambda_PML * grad(p), grad(v)) - pml.detJ * pml.k0**2 * inner(p, v)) * dx(2)

    # Assemble matrix
    a_form = form(a)
    A = assemble_matrix(a_form, bcs=[])
    A.assemble()

    # RHS vector
    ph = Function(V)
    ph.name = "p"
    jrwQb = 1j * rho_0 * omega * Q * b.x.petsc_vec
    
    # Solver
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.solve(jrwQb, ph.x.petsc_vec)
    ph.x.scatter_forward()

    # Submesh non-PML
    i_nopml = cell_tags.indices[(cell_tags.values == 1)]
    msh_nopml, entity_map, vertex_map, geom_map = create_submesh(msh, msh.topology.dim, i_nopml,)
    V_nopml = functionspace(msh_nopml, ("Lagrange", deg))
    ph_nopml = Function(V_nopml)
    ph_nopml.name = "p_nopml"

    # Interpolate to submesh
    cell_map = msh_nopml.topology.index_map(msh_nopml.topology.dim)
    num_cells_on_proc = cell_map.size_local + cell_map.num_ghosts
    interp_cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interp_data = create_interpolation_data(V_nopml, V, interp_cells, padding=1e-14)
    
    ph_nopml.interpolate_nonmatching(ph, interp_cells, interp_data)
    ph_nopml.x.scatter_forward()
    dx_nopml = Measure("dx", domain=msh_nopml)

    # Exact monopole
    def exact_solution(x):
        r = np.linalg.norm(x - x_S_exact[:, None], axis=0) 
        r = np.where(r < 1e-12, 1e-12, r)
        return 1j * rho_0 * omega * Q * np.exp(-1j * k0_val * r) / (4 * np.pi * r)
    
    # Calc exact and error
    p_exact = Function(V_nopml)
    p_exact.name = "p_exact"
    p_exact.interpolate(exact_solution)
    p_exact.x.scatter_forward()
    
    error = Function(V_nopml)
    error.name = "error_p"
    error.x.array[:] = p_exact.x.array - ph_nopml.x.array
    
    # Exclude singularity
    r_excl = 0.03
    coords = V_nopml.tabulate_dof_coordinates()
    dists = np.linalg.norm(coords - x_S_exact, axis=1)
    
    valid_dofs = dists >= r_excl
    
    # Max err
    local_max = np.max(np.abs(ph_nopml.x.array[valid_dofs] - p_exact.x.array[valid_dofs]))
    max_err = comm.allreduce(local_max, op=MPI.MAX)
    
    # L2 err
    x_ufl = ufl.SpatialCoordinate(msh_nopml)
    r_ufl = ufl.sqrt((x_ufl[0] - x_S_exact[0])**2 + (x_ufl[1] - x_S_exact[1])**2 + (x_ufl[2] - x_S_exact[2])**2)
    mask = ufl.conditional(ufl.ge(ufl.real(r_ufl), r_excl), 1.0, 0.0)

    L2_error_local = assemble_scalar(form(inner(error, error) * mask * dx_nopml))
    L2_error = np.sqrt(np.real(msh_nopml.comm.allreduce(L2_error_local, op=MPI.SUM)))

    L2_p_exact_local = assemble_scalar(form(inner(p_exact, p_exact) * mask * dx_nopml))
    L2_p_exact = np.sqrt(np.real(msh_nopml.comm.allreduce(L2_p_exact_local, op=MPI.SUM)))

    if L2_p_exact > 0:
        rel_L2_error = L2_error/L2_p_exact
    else:
        rel_L2_error = 0

    if comm.rank == 0:
        print(f"\nMax Abs Error:  {max_err:.4e}")
        print(f"Relative L2:    {rel_L2_error:.4%}")

    assert rel_L2_error < 5e-2, f"Relative L2 error too high: {rel_L2_error:.2%}"
    assert max_err < 5e-1, f"Max error exploded: {max_err:.2e}"