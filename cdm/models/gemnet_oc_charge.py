"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter, segment_coo

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_max_neighbors_mask,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base import BaseModel
from ocpmodels.modules.scaling.compat import load_scales_compat

from ocpmodels.models.gemnet_oc.initializers import get_initializer
from ocpmodels.models.gemnet_oc.interaction_indices import (
    get_mixed_triplets,
    get_quadruplets,
    get_triplets,
)
from ocpmodels.models.gemnet_oc.layers.atom_update_block import OutputBlock
from ocpmodels.models.gemnet_oc.layers.base_layers import Dense, ResidualLayer
from ocpmodels.models.gemnet_oc.layers.efficient import BasisEmbedding
from ocpmodels.models.gemnet_oc.layers.embedding_block import AtomEmbedding, EdgeEmbedding
from ocpmodels.models.gemnet_oc.layers.force_scaler import ForceScaler
from ocpmodels.models.gemnet_oc.layers.interaction_block import InteractionBlock
from ocpmodels.models.gemnet_oc.layers.radial_basis import RadialBasis
from ocpmodels.models.gemnet_oc.layers.spherical_basis import CircularBasisLayer, SphericalBasisLayer
from ocpmodels.models.gemnet_oc.utils import (
    get_angle,
    get_edge_id,
    get_inner_idx,
    inner_product_clamped,
    mask_neighbors,
    repeat_blocks,
)

from ocpmodels.models.gemnet_oc.gemnet_oc import GemNetOC


@registry.register_model("gemnet_oc_charge")
class GemNet_OC_charge(GemNetOC):

    def __init__(
        self,
        name='gemnet_oc_charge',
        num_spherical = 16,
        num_radial = 16,
        num_blocks = 6,
        emb_size_atom = 32,
        emb_size_edge = 32,
        emb_size_trip = 32,
        emb_size_rbf = 32,
        emb_size_cbf = 32,
        emb_size_sbf = 32,
        emb_size_trip_in = 32,
        emb_size_trip_out = 32,
        emb_size_quad_in = 32,
        emb_size_quad_out = 32,
        emb_size_aint_in = 32,
        emb_size_aint_out = 32,
        num_before_skip = 3,
        num_after_skip = 3,
        num_concat = 3,
        num_atom = 3,
        num_output_afteratom = 3,
        **kwargs,
    ):

        self.atomic = kwargs['atomic']
        self.probe = kwargs['probe']
        kwargs.pop('atomic')
        kwargs.pop('probe')
        
        if self.probe:
            kwargs['num_elements'] = 84
            
        kwargs['otf_graph'] = False
                 
        super().__init__(
            num_atoms = 1,
            bond_feat_dim = 1,
            num_targets = 1,
            num_spherical = num_spherical,
            num_radial = num_radial,
            num_blocks = num_blocks,
            emb_size_atom = emb_size_atom,
            emb_size_edge = emb_size_edge,
            emb_size_trip_in = emb_size_trip_in,
            emb_size_trip_out = emb_size_trip_out,
            emb_size_quad_in = emb_size_quad_in,
            emb_size_quad_out = emb_size_quad_out,
            emb_size_aint_in = emb_size_aint_in,
            emb_size_aint_out = emb_size_aint_out,
            emb_size_rbf = emb_size_rbf,
            emb_size_cbf = emb_size_cbf,
            emb_size_sbf = emb_size_sbf,
            num_before_skip = num_before_skip,
            num_after_skip = num_after_skip,
            num_concat = num_concat,
            num_atom = num_atom,
            num_output_afteratom = num_output_afteratom,
            **kwargs,
        )
        
        self.num_interactions = self.num_blocks
        
        self.hidden_channels = self.atom_emb.emb_size

        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        data.natoms = data.natoms.to(data.pos.device)
        data.tags = torch.ones_like(data.atomic_numbers)
        data.neighbors = data.neighbors.to(data.pos.device)
            
        (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        ) = self.get_graphs_and_indices(data)
        _, idx_t = main_graph["edge_index"]

        (
            basis_rad_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        ) = self.get_bases(
            main_graph=main_graph,
            a2a_graph=a2a_graph,
            a2ee2a_graph=a2ee2a_graph,
            qint_graph=qint_graph,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
            num_atoms=num_atoms,
        )


        # Embedding block
        if self.atomic:
            h = self.atom_emb(atomic_numbers)
        if self.probe:
            h = self.atom_emb(atomic_numbers + 1)
        # (nAtoms, emb_size_atom)
        
        m = self.edge_emb(h, basis_rad_raw, main_graph["edge_index"])
        # (nEdges, emb_size_edge)

        if self.atomic:
            h_list = []

            for i in range(self.num_blocks):
                # Interaction block
                h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases_qint,
                bases_e2e=bases_e2e,
                bases_a2e=bases_a2e,
                bases_e2a=bases_e2a,
                basis_a2a_rad=basis_a2a_rad,
                basis_atom_update=basis_atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
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
                bases_qint=bases_qint,
                bases_e2e=bases_e2e,
                bases_a2e=bases_a2e,
                bases_e2a=bases_e2a,
                basis_a2a_rad=basis_a2a_rad,
                basis_atom_update=basis_atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            return h[probe_indices]