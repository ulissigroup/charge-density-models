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

@registry.register_model("painn_charge")
class PaiNN_Charge(PaiNN):
    def __init__(
        self,
        name = 'painn_charge',
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
            otf_graph = False,
            **kwargs,
        )
        
        self.num_interactions = self.num_layers
        
        del self.out_energy
        del self.out_forces
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        data.pos = data.pos.float()
        pos = data.pos
        
        batch = data.batch
        z = data.atomic_numbers.long()
        
        data.natoms = data.natoms.to(data.cell.device)
        data.neighbors = data.neighbors.to(data.cell.device)
        
        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        assert z.dim() == 1 and z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope


        #### Interaction blocks ###############################################

        if self.atomic:
            x = self.atom_emb(z)
            vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
            
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
            x = self.atom_emb(z + 1) # +1 allows for probe embeddings
            vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
            
            atom_indices = torch.nonzero(z).flatten()
            probe_indices = (z == 0).nonzero().flatten()

            for i in range(self.num_layers):
                x[atom_indices] = data.atom_representations[i]
                
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
                
            return x[probe_indices]
