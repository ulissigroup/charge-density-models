"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from ocpmodels.common.registry import registry
import torch
from ocpmodels.models.painn.painn import PaiNN
from typing import Optional, Tuple

from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

import pdb

@registry.register_model("painn_charge")
class PaiNN_Charge(PaiNN):
    def __init__(
        self,
        use_pbc = True,
        regress_forces = False,
        direct_forces = False,
        otf_graph = False,
        
        hidden_channels = 128,
        num_interactions = 6,
        num_rbf = 32,
        cutoff = 5,
        max_neighbors = 200,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        scale_file: Optional[str] = None,
        num_elements = 83 + 1, #+1 for probe embedding
        
        atomic = False,
        probe = False,
        name = 'painn_charge',
    ):
        
        super().__init__(
            num_atoms = 1,
            bond_feat_dim = 1,
            num_targets = 1,
            hidden_channels = hidden_channels,
            num_layers = num_interactions,
            num_rbf = num_rbf,
            cutoff = cutoff,
            max_neighbors = max_neighbors,
            rbf = rbf,
            envelope = envelope,
            scale_file = scale_file,
            regress_forces = regress_forces,
            direct_forces = direct_forces,
            use_pbc = use_pbc,
            otf_graph = otf_graph,
            num_elements = num_elements,
        )
        self.atomic = atomic
        self.probe = probe
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        data.pos = data.pos.float()
        pos = data.pos
        
        batch = data.batch
        z = data.atomic_numbers.long()

        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        assert z.dim() == 1 and z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z + 1) # +1 to allow probe embeddings
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        #### Interaction blocks ###############################################

        if self.atomic:
            x_list = []
            for i in range(self.num_layers):
                dx, dvec = self.message_layers[i](
                    x, vec, edge_index, edge_rbf, edge_vector
                )
                
                x = x + dx
                vec = vec + dvec
                x = x * self.inv_sqrt_2
                
                dx, dvec = self.update_layers[i](x, vec)
                
                x = x + dx
                vec = vec + dvec
                x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)
                
                x_list.append(x)
                
            atom_representations = x_list
            return atom_representations
            
        if self.probe:
            atom_indices = torch.nonzero(z).flatten()
            probe_indices = (z == 0).nonzero().flatten()

            for i in range(self.num_layers):
                dx, dvec = self.message_layers[i](
                    x, vec, edge_index, edge_rbf, edge_vector
                )

                x = x + dx
                vec = vec + dvec
                x = x * self.inv_sqrt_2

                dx, dvec = self.update_layers[i](x, vec)

                x = x + dx
                vec = vec + dvec
                x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)
                
                x[atom_indices] = data.atom_representations[i]
                
            return x[probe_indices]
