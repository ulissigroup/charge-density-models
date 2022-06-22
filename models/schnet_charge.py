"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from ocpmodels.common.registry import registry
import torch
from ocpmodels.models.schnet import SchNetWrap as SchNet
from torch_scatter import scatter
import numpy as np

from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

import pdb

@registry.register_model("schnet_charge")
class schnet_charge(SchNet):
    def __init__(
        self,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout="add",
        atomic=False,
        probe=False,
        name='schnet_charge',
    ):

        super().__init__(
            num_atoms = 1,
            bond_feat_dim = 1,
            num_targets = 1,
            use_pbc=True,
            regress_forces=False,
            otf_graph=False,
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout
        )
        self.atomic = atomic
        self.probe = probe
        
        if self.probe:
            # From DeepDFT
            self.probe_state_gate_functions = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    ShiftedSoftplus(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.Sigmoid(),
                )
                for _ in range(num_interactions)
            ])

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        # TODO return distance computation in radius_graph_pbc to remove need
        # for get_pbc_distances call
        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )
            
            data.cell

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            edge_attr = self.distance_expansion(edge_weight)
            
            h = self.embedding(z)
            h[torch.sum(data.natoms):, :] = torch.zeros(*h[torch.sum(data.natoms):, :].size())

            if self.atomic:
                h_list = []
                for i, interaction, in enumerate(self.interactions):
                    h = h + interaction(h, edge_index, edge_weight, edge_attr)
                    h_list.append(h)
                    
                atom_representations = torch.stack(h_list)
                return atom_representations
            
            if self.probe:
                edge_weight = edge_weight.float()
                edge_attr = edge_attr.float()
                for interaction_number, (interaction, gate_layer) in enumerate(zip(self.interactions, self.probe_state_gate_functions)):
                    
                    msgsum = interaction(h, edge_index, edge_weight, edge_attr)
                    gates = gate_layer(h)
                
                    h = h * gates + (1-gates) * msgsum

                    h[:torch.sum(data.natoms), :] = data.atom_representations[interaction_number]

                return h[torch.sum(data.natoms):, :]
                
class ShiftedSoftplus(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x) - np.log(2.0)