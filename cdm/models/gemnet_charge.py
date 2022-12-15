"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import numpy as np
import torch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base import BaseModel
from ocpmodels.modules.scaling.compat import load_scales_compat

from ocpmodels.models.gemnet.layers.atom_update_block import OutputBlock
from ocpmodels.models.gemnet.layers.base_layers import Dense
from ocpmodels.models.gemnet.layers.efficient import EfficientInteractionDownProjection
from ocpmodels.models.gemnet.layers.embedding_block import AtomEmbedding, EdgeEmbedding
from ocpmodels.models.gemnet.layers.interaction_block import InteractionBlockTripletsOnly
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis
from ocpmodels.models.gemnet.layers.spherical_basis import CircularBasisLayer
from ocpmodels.models.gemnet.utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)

from ocpmodels.models.gemnet.gemnet import GemNetT


@registry.register_model("gemnet_t_charge")
class GemNetT_charge(GemNetT):

    def __init__(
        self,
        name='gemnet_t_charge',
        num_spherical = 16,
        num_radial = 16,
        num_blocks = 6,
        emb_size_atom = 64,
        emb_size_edge = 64,
        emb_size_trip = 64,
        emb_size_rbf = 64,
        emb_size_cbf = 64,
        emb_size_bil_trip = 64,
        num_before_skip = 3,
        num_after_skip = 3,
        num_concat = 3,
        num_atom = 3,
        **kwargs,
    ):

        self.atomic = kwargs['atomic']
        self.probe = kwargs['probe']
        kwargs.pop('atomic')
        kwargs.pop('probe')
        
        if self.probe:
            kwargs['num_elements'] = 84
                 
        super().__init__(
            num_atoms = 1,
            bond_feat_dim = 1,
            num_targets = 1,
            num_spherical = 16,
            num_radial = num_radial,
            num_blocks = num_blocks,
            emb_size_atom = emb_size_atom,
            emb_size_edge = emb_size_edge,
            emb_size_trip = emb_size_trip,
            emb_size_rbf = emb_size_rbf,
            emb_size_cbf = emb_size_cbf,
            emb_size_bil_trip = emb_size_bil_trip,
            num_before_skip = num_before_skip,
            num_after_skip = num_after_skip,
            num_concat = num_concat,
            num_atom = num_atom,
            otf_graph = False,
            **kwargs,
        )
        
        self.num_interactions = self.num_blocks
        
        self.hidden_channels = self.atom_emb.emb_size

        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        data.natoms = data.natoms.to(data.pos.device)
        data.neighbors = data.neighbors.to(data.pos.device)
            
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
        if self.atomic:
            h = self.atom_emb(atomic_numbers)
        if self.probe:
            h = self.atom_emb(atomic_numbers + 1)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        if self.atomic:
            h_list = []

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
                    idx_t=idx_t,
                )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
                
                h_list.append(h)
            return h_list
        
        if self.probe:
            
            atom_indices = torch.nonzero(data.atomic_numbers).flatten()
            probe_indices = (data.atomic_numbers == 0).nonzero().flatten()
            
            for i in range(self.num_blocks):
                h[atom_indices] = data.atom_representations[i]
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
                    idx_t=idx_t,
                )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            return h[probe_indices]