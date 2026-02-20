import os
import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import __version__ as dolfinx_version
from packaging.version import Version
if Version(dolfinx_version) > Version("0.9.0"):
    from dolfinx.io import gmsh as gmshio, VTXWriter
else: 
    from dolfinx.io import gmshio, VTXWriter

from dolfinx.fem import Function, functionspace, Constant
from ufl import ln, outer, Identity, grad, det, inv


class LcPML:
    def __init__(self, filename, d_pml, n_layers, comm=MPI.COMM_WORLD):
        self.filename = filename
        self.d_pml = d_pml
        self.n_layers = n_layers
        self.comm = comm
        self.rank = comm.rank
        
        self.mesh = None
        self.functions = {}
        
        self.surf_data = None
        self.meta = None

    def generate(self, physical_group):
        self.physical_group = physical_group
        if self.rank == 0:
            self._rank0_generate()
        
        self.surf_data = self.comm.bcast(self.surf_data, root=0)
        self.meta = self.comm.bcast(self.meta, root=0)
        
        mesh_data = gmshio.model_to_mesh(gmsh.model, self.comm, 0, gdim=3)
        if Version(dolfinx_version) > Version("0.9.0"):
            self.mesh = mesh_data.mesh
            self.cell_tags = mesh_data.cell_tags
            self.facet_tags = mesh_data.facet_tags
        else:
            self.mesh, self.cell_tags, self.facet_tags = mesh_data
        return self.mesh, self.cell_tags, self.facet_tags

    def compute_pml_properties(self):
        self.k0 = Constant(self.mesh, PETSc.ScalarType(1))
        V_s = functionspace(self.mesh, ("CG", 1))
        V_v = functionspace(self.mesh, ("CG", 1, (3,)))
        
        
        self.functions = {
            "n": Function(V_v, name="normal"),
            "csi": Function(V_s, name="pml_coordinate")
        }

        self._fill_local_data()
        self._compute_tensors()

    def export(self, out_filename="pml_output.bp"):
        if self.functions:
            with VTXWriter(self.comm, out_filename, list(self.functions.values()), engine="BP4") as vtx:
                vtx.write(0.0)

    def _rank0_generate(self):
        gmsh.initialize()
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.merge(self.filename)

        surf_ntags, coord = gmsh.model.mesh.getNodesForPhysicalGroup(2, self.physical_group)
        entities = gmsh.model.getEntitiesForPhysicalGroup(2, self.physical_group)
        
        faces = np.concatenate([gmsh.model.mesh.getElementFaceNodes(2, 3, e) for e in entities])
        sidx = np.argsort(surf_ntags)
        idx_sorted = np.searchsorted(surf_ntags, faces, sorter=sidx)

        V_raw = coord.reshape(-1, 3)
        F_raw = sidx[idx_sorted].reshape(-1, 3)

        v0, v1, v2 = V_raw[F_raw[:, 0]], V_raw[F_raw[:, 1]], V_raw[F_raw[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        
        n_raw = np.zeros_like(V_raw)
        np.add.at(n_raw, F_raw[:, 0], face_normals)
        np.add.at(n_raw, F_raw[:, 1], face_normals)
        np.add.at(n_raw, F_raw[:, 2], face_normals)
        
        n_raw /= np.linalg.norm(n_raw, axis=1)[:, np.newaxis]

        self.surf_data = {
            "surf_node_tags": surf_ntags.astype(np.int64), 
            "n": n_raw
        }
        self.meta = {"n_surf_nodes": len(surf_ntags), "start_vol_tag": gmsh.model.mesh.getMaxNodeTag() + 1}
        
        self._extrude(surf_ntags, V_raw, F_raw, n_raw)

    def _extrude(self, tags, coords, tris, norms):
        n_nodes = len(coords)
        dr = self.d_pml / self.n_layers
        
        sort_p = np.argsort(tris, axis=1)
        tris_s = np.take_along_axis(tris, sort_p, axis=1)
        is_odd = ((sort_p[:,1]-sort_p[:,0])*(sort_p[:,2]-sort_p[:,1])*(sort_p[:,2]-sort_p[:,0])) < 0

        start = self.meta["start_vol_tag"]
        new_tags = np.arange(start, start + self.n_layers * n_nodes, dtype=np.uint64)
        new_coords = np.zeros((self.n_layers * n_nodes, 3))
        tag_map = np.vstack([tags, new_tags.reshape(self.n_layers, n_nodes)])

        for L in range(1, self.n_layers + 1):
            new_coords[(L-1)*n_nodes : L*n_nodes] = coords + norms * (L * dr)

        tets = []
        for L in range(self.n_layers):
            b, t = tag_map[L, tris_s], tag_map[L+1, tris_s]
            
            layer_tets = [
                np.c_[b[:,0], b[:,1], b[:,2], t[:,2]],
                np.c_[b[:,0], b[:,1], t[:,2], t[:,1]],
                np.c_[b[:,0], t[:,1], t[:,2], t[:,0]]
            ]
            tets.extend([self._fix_tet_orientation(t, is_odd) for t in layer_tets])

        conn = np.vstack(tets).flatten().astype(np.uint64)
        pml_tag = gmsh.model.getEntities(3)[-1][1] + 1 if gmsh.model.getEntities(3) else 1
        
        max_elem_tag = gmsh.model.mesh.getMaxElementTag()
        n_new_elems = len(conn) // 4
        
        new_elem_tags = np.arange(max_elem_tag + 1, max_elem_tag + 1 + n_new_elems, dtype=np.uint64)
        
        gmsh.model.addDiscreteEntity(3, pml_tag)
        gmsh.model.mesh.addNodes(3, pml_tag, new_tags, new_coords.flatten())

        gmsh.model.mesh.addElements(3, pml_tag, [4], [new_elem_tags], [conn])
        gmsh.model.addPhysicalGroup(3, [pml_tag], -1, "PML_Domain")

    def _fix_tet_orientation(self, tet, odd_mask):
        fixed = tet.copy()
        fixed[odd_mask, 2], fixed[odd_mask, 3] = tet[odd_mask, 3], tet[odd_mask, 2]
        return fixed

    def _fill_local_data(self):
        local_tags = self.mesh.geometry.input_global_indices.astype(np.int64) + 1
        raw_tags = self.surf_data["surf_node_tags"]
        
        sorter = np.argsort(raw_tags)
        sorted_tags = raw_tags[sorter]
        
        is_surf = np.isin(local_tags, raw_tags)
        idx_s = np.where(is_surf)[0]
        
        start, count = self.meta["start_vol_tag"], self.meta["n_surf_nodes"]
        is_vol = (local_tags >= start) & (local_tags < start + self.n_layers * count)
        idx_v = np.where(is_vol)[0]

        pos = np.searchsorted(sorted_tags, local_tags[idx_s])
        raw_idx_s = sorter[pos]
        
        delta = local_tags[idx_v] - start
        raw_idx_v = delta % count 
        layers = (delta // count) + 1

        for name, fn in self.functions.items():
            arr = fn.x.array

            if name == "csi":
                arr[idx_v] = (layers / self.n_layers) * self.d_pml
                continue

            src_data = self.surf_data[name]
            arr[idx_s*3]   = src_data[raw_idx_s, 0]
            arr[idx_s*3+1] = src_data[raw_idx_s, 1]
            arr[idx_s*3+2] = src_data[raw_idx_s, 2]
            
            arr[idx_v*3]   = src_data[raw_idx_v, 0]
            arr[idx_v*3+1] = src_data[raw_idx_v, 1]
            arr[idx_v*3+2] = src_data[raw_idx_v, 2]
            
            fn.x.scatter_forward()

    def _compute_tensors(self):
        k0 = self.k0
        n = self.functions["n"]
        csi = self.functions["csi"]
        
        I = Identity(3)
        
        sigma = 1 / (self.d_pml - csi)
        f_csi = -ln(1 - csi / self.d_pml) 
        
        J_pml = I - (1 / (1j * k0)) * (sigma * outer(n, n) + f_csi * grad(n))
        
        self.detJ = det(J_pml)
        self.Lambda_PML = self.detJ * inv(J_pml) * inv(J_pml).T