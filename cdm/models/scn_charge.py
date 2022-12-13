"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pdb

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from ocpmodels.models.scn.scn import SphericalChannelNetwork as SCN

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)
from ocpmodels.models.scn.spherical_harmonics import SphericalHarmonicsHelper

try:
    import e3nn
    from e3nn import o3
except ImportError:
    pass


@registry.register_model("scn_charge")
class SCN_Charge(SCN):
    """Spherical Channel Network
    Paper: Spherical Channels for Modeling Atomic Interactions

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_num_neighbors (int): Maximum number of neighbors per atom
        cutoff (float):         Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_interactions (int): Number of layers in the GNN
        lmax (int):             Maximum degree of the spherical harmonics (1 to 10)
        mmax (int):             Maximum order of the spherical harmonics (0 or 1)
        num_resolutions (int):  Number of resolutions used to compute messages, further away atoms has lower resolution (1 or 2)
        sphere_channels (int):  Number of spherical channels
        sphere_channels_reduce (int): Number of spherical channels used during message passing (downsample or upsample)
        hidden_channels (int):  Number of hidden units in message passing
        num_taps (int):         Number of taps or rotations used during message passing (1 or otherwise set automatically based on mmax)

        use_grid (bool):        Use non-linear pointwise convolution during aggregation
        num_bands (int):        Number of bands used during message aggregation for the 1x1 pointwise convolution (1 or 2)

        num_sphere_samples (int): Number of samples used to approximate the integration of the sphere in the output blocks
        num_basis_functions (int): Number of basis functions used for distance and atomic number blocks
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        basis_width_scalar (float): Width of distance basis function
        distance_resolution (float): Distance between distance basis functions in Angstroms

        show_timing_info (bool): Show timing and memory info
    """

    def __init__(
        self,
        name = 'scn_charge',
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


    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.device = data.pos.device
        self.num_atoms = len(data.batch)
        self.batch_size = len(data.natoms)
        # torch.autograd.set_detect_anomaly(True)

        start_time = time.time()

        outputs = self._forward_helper(
            data,
        )

        if self.show_timing_info is True:
            torch.cuda.synchronize()
            print(
                "{} Time: {}\tMemory: {}\t{}".format(
                    self.counter,
                    time.time() - start_time,
                    len(data.pos),
                    torch.cuda.max_memory_allocated() / 1000000,
                )
            )

        self.counter = self.counter + 1

        return outputs

    # restructure forward helper for conditional grad
    def _forward_helper(self, data):
        atomic_numbers = data.atomic_numbers.long()
        
        num_atoms = len(atomic_numbers)
        pos = data.pos

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################
        
        # Calculate which message block each edge should use. Based on edge distance rank.
        edge_rank = self._rank_edge_distances(
            edge_distance, edge_index, self.max_num_neighbors,
        )

        # Reorder edges so that they are grouped by distance rank (lowest to highest)
        last_cutoff = -0.1
        message_block_idx = torch.zeros(len(edge_distance), device=pos.device)
        edge_distance_reorder = torch.tensor([], device=self.device)
        edge_index_reorder = torch.tensor([], device=self.device)
        edge_distance_vec_reorder = torch.tensor([], device=self.device)
        cutoff_index = torch.tensor([0], device=self.device)
        for i in range(self.num_resolutions):
            mask = torch.logical_and(
                edge_rank.gt(last_cutoff), edge_rank.le(self.cutoff_list[i])
            )
            last_cutoff = self.cutoff_list[i]
            message_block_idx.masked_fill_(mask, i)
            edge_distance_reorder = torch.cat(
                [
                    edge_distance_reorder,
                    torch.masked_select(edge_distance, mask),
                ],
                dim=0,
            )
            edge_index_reorder = torch.cat(
                [
                    edge_index_reorder,
                    torch.masked_select(
                        edge_index, mask.view(1, -1).repeat(2, 1)
                    ).view(2, -1),
                ],
                dim=1,
            )
            edge_distance_vec_mask = torch.masked_select(
                edge_distance_vec, mask.view(-1, 1).repeat(1, 3)
            ).view(-1, 3)
            edge_distance_vec_reorder = torch.cat(
                [edge_distance_vec_reorder, edge_distance_vec_mask], dim=0
            )
            cutoff_index = torch.cat(
                [
                    cutoff_index,
                    torch.tensor(
                        [len(edge_distance_reorder)], device=self.device
                    ),
                ],
                dim=0,
            )

        edge_index = edge_index_reorder.long()
        edge_distance = edge_distance_reorder
        edge_distance_vec = edge_distance_vec_reorder

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.sphharm_list[i].InitWignerDMatrix(
                edge_rot_mat[cutoff_index[i] : cutoff_index[i + 1]],
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = torch.zeros(
            num_atoms,
            self.sphere_basis,
            self.sphere_channels,
            device=pos.device,
        )
        x[:, 0, :] = self.sphere_embedding(atomic_numbers)

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        
        if self.atomic:
            atom_representations = []
            for i, interaction in enumerate(self.edge_blocks):
                if i > 0:
                    x = x + interaction(
                        x, atomic_numbers, edge_distance, edge_index, cutoff_index
                    )
                    atom_representations.append(x)
                else:
                    x = interaction(
                        x, atomic_numbers, edge_distance, edge_index, cutoff_index
                    )
                    atom_representations.append(x)
            return atom_representations
        
        
        if self.probe:
            atom_indices = torch.nonzero(data.atomic_numbers).flatten()
            probe_indices = (data.atomic_numbers == 0).nonzero().flatten()
            
            for i, interaction in enumerate(self.edge_blocks):
                if i > 0:
                    x = x + interaction(
                        x, atomic_numbers, edge_distance, edge_index, cutoff_index
                    )
                    x[atom_indices] = data.atom_representations[i]
                else:
                    x = interaction(
                        x, atomic_numbers, edge_distance, edge_index, cutoff_index
                    )
                    x[atom_indices] = data.atom_representations[i]
                    
            ###############################################################
            # Predict electron density
            ###############################################################
            
             # Create a roughly evenly distributed point sampling of the sphere
            sphere_points = CalcSpherePoints(
                self.num_sphere_samples, x.device
            ).detach()
            sphharm_weights = o3.spherical_harmonics(
                torch.arange(0, self.lmax + 1).tolist(), sphere_points, False
            ).detach()

            # Density estimation
            node_energy = torch.einsum(
                "abc, pb->apc", x, sphharm_weights
            ).contiguous()
            node_energy = node_energy.view(-1, self.sphere_channels)
            node_energy = self.act(self.energy_fc1(node_energy))
            node_energy = self.act(self.energy_fc2(node_energy))
            node_energy = self.energy_fc3(node_energy)
            node_energy = node_energy.view(-1, self.num_sphere_samples, 1)
            node_density = torch.sum(node_energy, dim=1) / self.num_sphere_samples

            return node_density[probe_indices]
    
    def _rank_edge_distances(
        self, edge_distance, edge_index, max_num_neighbors
    ):
        
        device = edge_distance.device
        
        # Create an index map to map distances from atom_distance to distance_sort
        # index_sort_map assumes index to be sorted
        output, num_neighbors = torch.unique(edge_index[1], return_counts=True)
        
        index_neighbor_offset = (
            torch.cumsum(num_neighbors, dim=0) - num_neighbors
        )
        index_neighbor_offset_expand = torch.repeat_interleave(
            index_neighbor_offset, num_neighbors
        )

        index_sort_map = (
            edge_index[1] * max_num_neighbors
            + torch.arange(len(edge_distance), device=device)
            - index_neighbor_offset_expand
        )

        num_atoms = torch.max(edge_index) + 1
        distance_sort = torch.full(
            [num_atoms * max_num_neighbors], np.inf, device=device
        )
        distance_sort.index_copy_(0, index_sort_map, edge_distance)
        distance_sort = distance_sort.view(num_atoms, max_num_neighbors)
        no_op, index_sort = torch.sort(distance_sort, dim=1)

        index_map = (
            torch.arange(max_num_neighbors, device=device)
            .view(1, -1)
            .repeat(num_atoms, 1)
            .view(-1)
        )
        index_sort = index_sort + (
            torch.arange(num_atoms, device=device) * max_num_neighbors
        ).view(-1, 1).repeat(1, max_num_neighbors)
        edge_rank = torch.zeros_like(index_map)
        edge_rank.index_copy_(0, index_sort.view(-1), index_map)
        edge_rank = edge_rank.view(num_atoms, max_num_neighbors)

        index_sort_mask = distance_sort.lt(1000.0)
        edge_rank = torch.masked_select(edge_rank, index_sort_mask)

        return edge_rank
    

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
