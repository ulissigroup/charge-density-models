"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import numpy as np
from typing import Optional
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.gemnet.utils import inner_product_normalized
from ocpmodels.models.gemnet.gemnet import GemNetT
from ocpmodels.common.registry import registry

'''
import numpy as np
import torch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from torch_sparse import SparseTensor
import pdb
'''

'''
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
'''

'''
from ocpmodels.models.gemnet.layers.atom_update_block import OutputBlock
from ocpmodels.models.gemnet.layers.base_layers import Dense
from ocpmodels.models.gemnet.layers.efficient import EfficientInteractionDownProjection
from ocpmodels.models.gemnet.layers.embedding_block import AtomEmbedding, EdgeEmbedding
from ocpmodels.models.gemnet.layers.interaction_block import InteractionBlockTripletsOnly
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis
from ocpmodels.models.gemnet.layers.scaling import AutomaticFit
from ocpmodels.models.gemnet.layers.spherical_basis import CircularBasisLayer
'''

@registry.register_model("gemnet_charge")
class GemNetT_charge(GemNetT):
    def __init__(
        self,
        scale_file: str,
        num_atoms: int = 1,
        bond_feat_dim: int = 1,
        num_targets: int = 1,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_interactions: int = 3,
        hidden_channels: int = 512,
        emb_size_edge: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 64,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        regress_forces: bool = True,
        direct_forces: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        extensive: bool = True,
        otf_graph: bool = False,
        use_pbc: bool = True,
        output_init: str = "HeOrthogonal",
        activation: str = "swish", 
        atomic: bool = False,
        probe: bool = False,
        name: str = 'gemnet_charge',
    ):
        num_atoms = 1
        super().__init__(
            num_atoms = num_atoms,
            bond_feat_dim = bond_feat_dim,
            num_targets = num_targets,
            num_spherical = num_spherical,
            num_radial = num_radial,
            num_blocks = num_interactions,
            emb_size_atom = hidden_channels,
            emb_size_edge = emb_size_edge,
            emb_size_trip = emb_size_trip,
            emb_size_rbf = emb_size_rbf,
            emb_size_cbf = emb_size_cbf,
            emb_size_bil_trip = emb_size_bil_trip,
            num_before_skip = num_before_skip,
            num_after_skip = num_after_skip,
            num_concat = num_concat,
            num_atom = num_atom,
            regress_forces = regress_forces,
            direct_forces = direct_forces,
            cutoff = cutoff,
            max_neighbors = max_neighbors,
            rbf = rbf,
            envelope = envelope,
            cbf = cbf,
            extensive = extensive,
            otf_graph = otf_graph,
            use_pbc = use_pbc,
            output_init = output_init,
            activation = activation,
            scale_file = scale_file,
        )
        
        self.atomic=atomic
        self.probe=probe
        
        self.atom_emb = AtomAndProbeEmbedding(emb_size = hidden_channels)
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        batch = data.batch
        pos = data.pos
        atomic_numbers = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data)
        idx_s, idx_t = edge_index

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        # (nAtoms, num_targets), (nEdges, num_targets)
        
        
        if self.atomic:   
            h_list = []
            m_list = []
            for i in range(self.num_blocks):
                # Interaction block
                h, m = self.int_blocks[i](
                    h=h,
                    m=m,
                    rbf3=rbf3,
                    cbf3=cbf3,
                    id3_ragged_idx=id3_ragged_idx,
                    id_swap=id_swap,
                    id3_ba=id3_ba,
                    id3_ca=id3_ca,
                    rbf_h=rbf_h,
                    idx_s=idx_s,
                    idx_t=idx_t
                )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

                h_list.append(h)
                m_list.append(m)


            data.atom_representations = h_list
            data.edge_representations = m_list
            
            data.atom_representations = torch.stack(data.atom_representations)
            data.edge_representations = torch.stack(data.edge_representations)
            
            return data
            
        if self.probe:
            for interaction_number, interaction in enumerate(self.int_blocks):
                # Interaction block
                h, m = interaction(
                    h=h,
                    m=m,
                    rbf3=rbf3,
                    cbf3=cbf3,
                    id3_ragged_idx=id3_ragged_idx,
                    id_swap=id_swap,
                    id3_ba=id3_ba,
                    id3_ca=id3_ca,
                    rbf_h=rbf_h,
                    idx_s=idx_s,
                    idx_t=idx_t
                )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
                h[:torch.sum(data.natoms), :] = data.atom_representations[interaction_number]
                m[:torch.sum(data.num_atomic_edges), :] = data.edge_representations[interaction_number]
                
            return h[torch.sum(data.natoms):, :]
        
class AtomAndProbeEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type
    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Bi (83) plus 1 for probe points.
        self.embeddings = torch.nn.Embedding(83 + 1, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(
            self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z)  # Z=0: probe point
        return h