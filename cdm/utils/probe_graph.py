import torch
import numpy as np

from torch_geometric.data import Data

from ase import Atoms
from ase.calculators.vasp import VaspChargeDensity
from ase import neighborlist as nbl

from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import data_list_collater

class ProbeGraphAdder():
    '''
    A class that is used to add probe graphs to data objects.
    The data object must have an attribute "charge_density" which is
    a 3-dimensional tensor of charge density values
    '''
    def __init__(self, 
                 num_probes=1000, 
                 cutoff=5, 
                 include_atomic_edges=False, 
                 mode = 'random', 
                 slice_start = 0,
                 stride = 1,
                 implementation = 'RGPBC',
                ):
        self.num_probes = num_probes
        self.cutoff = cutoff
        self.include_atomic_edges = include_atomic_edges
        self.mode = mode
        self.slice_start = slice_start
        self.stride = stride
        self.implementation = implementation

        
    def __call__(
        self,
        data_object, 
        slice_start = None,
        specify_probes = None,
        num_probes = None,
        mode = None,
        stride = None,
        use_tqdm = False,
    ):
        
        if self.implementation == 'SKIP':
            return data_object
        
        # Check if probe graph has been precomputed
        if hasattr(data_object, 'probe_data'):
            if hasattr(data_object.probe_data, 'edge_index') and hasattr(data_object.probe_data, 'cell_offsets'):
                return data_object

        # Handle batching
        if type(data_object.natoms) is not int:
            if len(data_object.natoms) > 1:
                data_list = data_object.to_data_list()
                batches = [data_list_collater([data]) for data in data_list]
                probe_data = [self(batch).probe_data for batch in batches]
                probe_data = data_list_collater(probe_data)
                data_object.probe_data = probe_data
                return data_object
        
        # Use default options if none have been passed in
        if slice_start is None:
            slice_start = self.slice_start
        if num_probes is None:
            num_probes = self.num_probes
        if mode is None:
            mode = self.mode
        if stride is None:
            stride = self.stride
        
        probe_data = Data()
        density = torch.tensor(data_object.charge_density)

        if stride != 1:
            assert (stride == 2) or (stride == 4)
            density = density[::stride, ::stride, ::stride]

        grid_pos = calculate_grid_pos(density.shape, data_object.cell)

        if mode == 'random':
            probe_choice = np.random.randint(np.prod(grid_pos.shape[-5:-2]), size = num_probes)
            probe_choice = np.unravel_index(probe_choice, grid_pos.shape[-5:-2])

            out = get_edges_from_choice(
                probe_choice,
                grid_pos,
                atom_pos = data_object.pos,
                cell = data_object.cell,
                cutoff = self.cutoff,
                include_atomic_edges = self.include_atomic_edges,
                implementation = self.implementation,
            )
            probe_edges, probe_offsets, probe_pos = out
            atomic_numbers = torch.clone(data_object.atomic_numbers.detach())
            atomic_numbers = torch.cat((atomic_numbers, torch.zeros(num_probes, device = atomic_numbers.device)))
            
        elif mode == 'specify':
            num_probes = len(specify_probes)
            probe_choice = np.unravel_index(specify_probes, grid_pos.shape[-5:-2])
            out = get_edges_from_choice(
                probe_choice,
                grid_pos,
                atom_pos = data_object.pos,
                cell = data_object.cell,
                cutoff = self.cutoff,
                include_atomic_edges = self.include_atomic_edges,
                implementation = self.implementation,
            )
            probe_edges, probe_offsets, probe_pos = out
            atomic_numbers = torch.cat((data_object.atomic_numbers, torch.zeros(num_probes, device = data_object.atomic_numbers.device)))

        elif mode == 'slice':
            probe_choice = np.arange(slice_start, slice_start + num_probes, step=1)
            probe_choice = np.unravel_index(probe_choice, grid_pos.shape[-5:-2])
            out = get_edges_from_choice(
                probe_choice,
                grid_pos,
                atom_pos = data_object.pos,
                cell = data_object.cell,
                cutoff = self.cutoff,
                include_atomic_edges = self.include_atomic_edges,
                implementation = self.implementation,
            )
            probe_edges, probe_offsets, probe_pos = out
            atomic_numbers = torch.cat((data_object.atomic_numbers, torch.zeros(num_probes, device = data_object.atomic_numbers.device)))
            
        elif mode == 'all':
            total_probes = np.prod(density.shape)
            num_blocks = int(np.ceil(total_probes / num_probes))
            
            probe_edges = torch.tensor([], device = data_object.edge_index.device)
            probe_offsets = torch.tensor([], device = data_object.cell_offsets.device)
            atomic_numbers = torch.clone(data_object.atomic_numbers.detach())
            probe_pos = torch.tensor([], device = data_object.pos.device)
            
            loop = range(num_blocks)
            if use_tqdm:
                loop = tqdm(loop)

            for i in loop:
                if i == num_blocks - 1:
                    probe_choice = np.arange(i * num_probes,  total_probes, step = 1)
                else:
                    probe_choice = np.arange(i * num_probes, (i+1)*num_probes, step = 1)
                    
                probe_choice = np.unravel_index(probe_choice, grid_pos.shape[-5:-2])
                out = get_edges_from_choice(
                    probe_choice,
                    grid_pos,
                    atom_pos = data_object.pos,
                    cell = data_object.cell,
                    cutoff = self.cutoff,
                    include_atomic_edges = self.include_atomic_edges,
                    implementation = self.implementation,
                )
                new_edges, new_offsets, new_pos = out
                
                new_edges[1] += i*num_probes
                probe_edges = torch.cat((probe_edges, new_edges), dim=1)
                probe_offsets = torch.cat((probe_offsets, new_offsets))
                atomic_numbers = torch.cat((atomic_numbers, torch.zeros(new_pos.shape[0], device = atomic_numbers.device)))
                probe_pos = torch.cat((probe_pos, new_pos))
                
            probe_choice = np.arange(0, np.prod(grid_pos.shape[-5:-2]), step=1)
            probe_choice = np.unravel_index(probe_choice, grid_pos.shape[-5:-2])
            
        else:
            raise RuntimeError('Mode '+mode+' is not recognized.')
        
        # Add attributes to probe_data object
        probe_data.cell = data_object.cell
        probe_data.atomic_numbers = atomic_numbers
        probe_data.natoms = torch.LongTensor([int(len(atomic_numbers))])
        probe_data.pos = torch.cat((data_object.pos, probe_pos))

        probe_data.target = torch.tensor(density.reshape(density.shape[-3:])[probe_choice[0:3]], device = probe_pos.device)
        
        probe_data.edge_index = probe_edges.long()
        
        probe_data.cell_offsets = -probe_offsets
        
        probe_data.neighbors = torch.LongTensor([probe_data.edge_index.shape[1]])

        probe_data.total_target = torch.sum(density) * torch.numel(probe_data.target) / torch.numel(density)
        
        # Add probe_data object to overall data object
        data_object.probe_data = probe_data

        return data_object
    
class AseNeighborListWrapper:
    """
    Wrapper around ASE neighborlist
    Modified from DeepDFT
    """

    def __init__(self, cutoff, atom_pos, probe_pos, cell):
        atoms = Atoms(numbers = [1] * len(atom_pos),
        positions = atom_pos.cpu().detach().numpy(),
        cell = cell.cpu().detach().numpy()[0],
        pbc = [True, True, True])
        
        probe_atoms = Atoms(numbers = [0] * len(probe_pos), positions = probe_pos)
        atoms_with_probes = atoms.copy()
        atoms_with_probes.extend(probe_atoms)
        
        atoms = atoms_with_probes
        
        self.neighborlist = nbl.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()
        
        is_probe = atoms.get_atomic_numbers() == 0
        self.num_atoms = len(atoms.get_positions()[~is_probe])
        self.atomic_numbers = atoms.get_atomic_numbers()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"
        
        indices, offsets = self.neighborlist.get_neighbors(i)
         
        offsets = offsets
        
        return indices, offsets
    
    def get_all_neighbors(self, cutoff, include_atomic_edges):
        probe_edges = []
        probe_offsets = []
        results = [self.neighborlist.get_neighbors(i) for i in range(self.num_atoms)]
        
        for i, (neigh_idx, neigh_offset) in enumerate(results):
            if not include_atomic_edges:
                neigh_atomic_species = self.atomic_numbers[neigh_idx]
                neigh_is_probe = neigh_atomic_species == 0
                neigh_idx = neigh_idx[neigh_is_probe]
                neigh_offset = neigh_offset[neigh_is_probe]
            
            atom_index = np.ones_like(neigh_idx) * i
            edges = np.stack((atom_index, neigh_idx), axis = 1)
            probe_edges.append(edges)
            probe_offsets.append(neigh_offset)
        
        edge_index = torch.tensor(np.concatenate(probe_edges, axis=0)).T
        
        cell_offsets = torch.tensor(np.concatenate(probe_offsets, axis=0))
        
        return edge_index, cell_offsets
    
class RadiusGraphPBCWrapper:
    """
    Wraps a modified version of the neighbor-finding algorithm from ocp
    (ocp.ocpmodels.common.utils.radius_graph_pbc)
    The modifications restrict the neighbor-finding to atom-probe edges,
    which is more efficient for our purposes.
    """
    def __init__(self, radius, atom_pos, probe_pos, cell, pbc = [True, True, False]):
        self.cutoff = radius
        atom_indices = torch.arange(0, len(atom_pos), device = atom_pos.device)
        probe_indices = torch.arange(len(atom_pos), len(atom_pos)+len(probe_pos), device = probe_pos.device)
        batch_size = 1
        
        num_atoms = len(atom_pos)
        num_probes = len(probe_pos)
        num_total = num_atoms + num_probes
        num_combos = num_atoms * num_probes
        
        indices = np.arange(0, num_total, 1)

        index1 = torch.repeat_interleave(atom_indices, repeats=num_probes)
        index2 = probe_indices.repeat(num_atoms)

        pos1 = atom_pos[index1]
        pos2 = probe_pos[index2 - num_atoms]

        cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
        cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
        
        if pbc[0]:
            inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
            rep_a1 = torch.ceil(radius * inv_min_dist_a1)
        else:
            rep_a1 = cell.new_zeros(1)
        
        if pbc[1]:
            cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
            inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
            rep_a2 = torch.ceil(radius * inv_min_dist_a2)
        else:
            rep_a2 = cell.new_zeros(1)
        
        if pbc[2]:
            cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
            inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
            rep_a3 = torch.ceil(radius * inv_min_dist_a3)
        else:
            rep_a3 = cell.new_zeros(1)

        # Take the max over all images for uniformity. This is essentially padding.
        # Note that this can significantly increase the number of computed distances
        # if the required repetitions are very different between images
        # (which they usually are). Changing this to sparse (scatter) operations
        # might be worth the effort if this function becomes a bottleneck.
        max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]
        
        # Tensor of unit cells
        cells_per_dim = [
            torch.arange(-rep, rep + 1, dtype=torch.float, device = cell.device)
            for rep in max_rep
        ]
        unit_cell = torch.cartesian_prod(*cells_per_dim)
        
        num_cells = len(unit_cell)
        unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
            len(index2), 1, 1
        )
        unit_cell = torch.transpose(unit_cell, 0, 1)
        unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
            batch_size, -1, -1
        )

        # Compute the x, y, z positional offsets for each cell in each image
        data_cell = torch.transpose(cell, 1, 2)
        pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
        pbc_offsets_per_atom = torch.repeat_interleave(
            pbc_offsets, num_combos, dim=0
        )

        # Expand the positions and indices for the 9 cells
        pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
        pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
        index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
        index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
        # Add the PBC offsets for the second atom
        pos2 = pos2 + pbc_offsets_per_atom

        # Compute the squared distance between atoms
        atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
        atom_distance_sqr = atom_distance_sqr.view(-1)

        # Remove pairs that are too far apart
        mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
        
        # Remove pairs with the same atoms (distance = 0.0)
        mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
        mask = torch.logical_and(mask_within_radius, mask_not_same)
        index1 = torch.masked_select(index1, mask)
        index2 = torch.masked_select(index2, mask)
        
        unit_cell = torch.masked_select(
            unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)
        atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

        self.edge_index = torch.stack((index1, index2))

        self.offsets = unit_cell
        
    def get_all_neighbors(self, cutoff, include_atomic_edges = False):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"
        
        if include_atomic_edges:
            raise NotImplementedError
        
        return self.edge_index.to(torch.int64), self.offsets
    
def calculate_grid_pos(shape, cell):
    # Ensure proper dimensions of cell
    if len(cell.shape) > 2:
        if cell.shape[0] == 1:
            cell = cell[0, :, :]
        else:
            raise NotImplementedError('calculate_grid_pos does not yet support batch sizes > 1')
    else:
        raise RuntimeError('Invalid unit cell definition for calculate_grid_pos')
    
    # Compute grid positions
    grid_pos = torch.cartesian_prod(
        torch.linspace(0, 1, shape[-3]+1, device = cell.device)[:-1],
        torch.linspace(0, 1, shape[-2]+1, device = cell.device)[:-1],
        torch.linspace(0, 1, shape[-1]+1, device = cell.device)[:-1],
    )
    
    grid_pos = torch.mm(grid_pos, cell)
    grid_pos = grid_pos.reshape((*shape, 1, 3))
    return grid_pos

def get_edges_from_choice(probe_choice, grid_pos, atom_pos, cell, cutoff, include_atomic_edges, implementation):
        """
        Given a list of chosen probes, compute all edges between the probes and atoms.
        Portions from DeepDFT
        """ 
        grid_pos = grid_pos.reshape((*grid_pos.shape[-5:-2], 3))
        probe_pos = grid_pos[probe_choice[0:3]]
        
        if implementation == 'ASE':
            neighborlist = AseNeighborListWrapper(cutoff, atom_pos, probe_pos, cell)
        
        elif implementation == 'RGPBC':
            neighborlist = RadiusGraphPBCWrapper(cutoff, atom_pos, probe_pos, cell)
            
        else:
            raise NotImplementedError('Unsupported implementation. Please choose from: ASE, RGPBC')
        
        edge_index, cell_offsets = neighborlist.get_all_neighbors(cutoff, include_atomic_edges)

        return edge_index, cell_offsets, probe_pos
