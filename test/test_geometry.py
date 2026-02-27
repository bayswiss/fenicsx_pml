import numpy as np
import pytest
from mpi4py import MPI
from fenicsx_pml import LcPML

@pytest.fixture
def pml_setup():
    """Fixture to generate the PML once per test session."""
    pml = LcPML("polyhedron.msh", d_pml=0.03, n_layers=3, comm=MPI.COMM_WORLD)
    msh, _ , __ = pml.generate(physical_group=3)
    pml.compute_pml_properties()
    pml.export()
    return pml, msh

def test_csi_bounds(pml_setup):
    pml, msh = pml_setup
    csi_array = pml.functions["csi"].x.array
    
    # Check bounds exactly. Tolerance handles floating point precision.
    assert np.isclose(np.min(csi_array), 0.0, atol=1e-10), "PML csi does not start at 0.0"
    assert np.isclose(np.max(csi_array), pml.d_pml, atol=1e-10), "PML csi does not end at d_pml"

def test_normal_magnitudes(pml_setup):
    pml, msh = pml_setup
    n_array = pml.functions["n"].x.array
    
    # Reshape the flat array into (N_nodes, 3) vectors
    vectors = n_array.reshape(-1, 3)
    magnitudes = np.linalg.norm(vectors, axis=1)
    
    # Create a mask to only test nodes that actually have a normal vector (PML nodes)
    non_zero_mask = magnitudes > 1e-8
    
    assert np.allclose(magnitudes[non_zero_mask], 1.0), "Normal vectors are not correctly normalized to 1.0"