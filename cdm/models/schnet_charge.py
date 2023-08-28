"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from ocpmodels.common.registry import registry
import torch
from ocpmodels.models.schnet import SchNetWrap as SchNet

from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

@registry.register_model("schnet_charge")
class schnet_charge(SchNet):
    def __init__(
        self,
        name='schnet_charge',
        **kwargs,
    ):

        self.atomic = kwargs['atomic']
        self.probe = kwargs['probe']
        kwargs.pop('atomic')
        kwargs.pop('probe')
                 
        super().__init__(
            num_atoms = 1,
            bond_feat_dim = 1,
            num_targets = 1,
            otf_graph = False,
            **kwargs,
        )
        
        if hasattr(self, 'lin1'):
            del self.lin1
        if hasattr(self, 'lin2'):
            del self.lin2
            
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        
        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long
            
        edge_attr = self.distance_expansion(edge_weight)
            
        h = self.embedding(z)

        if self.atomic:
            h_list = []
            for i, interaction, in enumerate(self.interactions):
                h = h + interaction(h, edge_index, edge_weight, edge_attr)
                h_list.append(h)

            atom_representations = h_list
            return atom_representations

        if self.probe:
            atom_indices = torch.nonzero(data.atomic_numbers).flatten()
            probe_indices = (data.atomic_numbers == 0).nonzero().flatten()
            
            edge_weight = edge_weight.float()
            edge_attr = edge_attr.float()
            
            for interaction_number, interaction in enumerate(self.interactions):
                h[atom_indices] = data.atom_representations[interaction_number]
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            return h[probe_indices]
