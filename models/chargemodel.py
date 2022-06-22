import torch
from chg_utils import get_probe_graph
from ocpmodels.common.utils import conditional_grad
from ocpmodels.common.registry import registry
from ocpmodels.datasets import data_list_collater
from ase import Atoms
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import pdb
from tqdm import tqdm

@registry.register_model("charge_model")
class ChargeModel(torch.nn.Module):
    def __init__(
        self,
        atom_model_config,
        probe_model_config,
        probe_state_size,
        include_atomic_edges = False,
        name = 'charge_model',
        atom_representation_reduction = None,
    ):
        super().__init__()
        self.regress_forces = False
        
        self.probe_output_function = torch.nn.Sequential(
            torch.nn.Linear(probe_state_size, probe_state_size),
            torch.nn.ELU(),
            torch.nn.Linear(probe_state_size, 1),
            torch.nn.ELU(),
        )

        
        self.atom_message_model = registry.get_model_class(atom_model_config['name'])(**atom_model_config, atomic=True, probe=False)
        self.probe_message_model = registry.get_model_class(probe_model_config['name'])(**probe_model_config, atomic=False, probe=True)
        
        if atom_model_config['name'] == 'gemnet_charge' or probe_model_config['name'] == 'gemnet_charge':
            self.include_atomic_edges = True
        else:
            self.include_atomic_edges = False
            
        if atom_representation_reduction is not None:
            self.reduce_atom_representations = True
            self.atom_reduction = torch.nn.Sequential(
                torch.nn.Linear(atom_representation_reduction[0], atom_representation_reduction[0]),
                torch.nn.Sigmoid(),
                torch.nn.Linear(atom_representation_reduction[0], atom_representation_reduction[1]))
        else:
            self.reduce_atom_representations = False
        
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # Ensure data has probe points
        #assert hasattr(data, 'charge_edges')
        #assert hasattr(data, 'charge_pos')
        #assert hasattr(data, 'charge_cell_offsets')
        #assert hasattr(data, 'charge_neighbors')
        #assert hasattr(data, 'charge_density')
        
        #data.charge_edges = data.charge_edges.flipud()
        
        atom_representations = self.forward_atomic(data)
        
        #probe_graph = get_probe_graph(data)
    
        probes = self.forward_probe(data, atom_representations)
        
        return probes
        
        '''
        data = self.atom_message_model(data)
        
        if self.reduce_atom_representations:
            data.atom_representations = self.atom_reduction(data.atom_representations)
            
        atoms_objects = self.batch_to_atoms(data)
        probe_representations = []
        offset = torch.sum(data.natoms)
        
        for i, atoms_object in enumerate(atoms_objects):
            struct = AseAtomsAdaptor.get_structure(atoms_object)
            density = data.charge_density[i]
            probes_sampled = 0
            n_probes = density.shape

            probe_index = np.mgrid[0:n_probes[0], 0:n_probes[1], 0:n_probes[2]].reshape(3, -1).T
            
            counter = 0

            while probes_sampled < np.prod(n_probes):
                
                counter += 1
                print(counter)
                
                if probes_sampled + self.chunk_size < np.prod(n_probes):
                    next_size = self.chunk_size
                else:
                    next_size = np.prod(n_probes) - probes_sampled

                probe_subset = probe_index[probes_sampled:probes_sampled+next_size]
                site_list = []
                charge_pos = np.zeros(probe_subset.shape)
                uc_coords = probe_subset / n_probes
                
                charge_pos = PeriodicSite(
                                species = 1,
                                coords = uc_coords,
                                lattice = struct.lattice,
                                to_unit_cell = True
                            ).coords
                
                site_list = [PeriodicSite(
                        species = 1,
                        coords = coords,
                        lattice = struct.lattice,
                        to_unit_cell = False)
                             for coords in charge_pos]
                    
                a, b, c, d = struct.get_neighbor_list(
                    r=self.probe_cutoff,
                    sites = site_list,
                    numerical_tol=0, 
                    exclude_self=True
                )

                edge_index, edge_distances, cell_offsets = self._reshape_features(
                    a, b, d, c)
                
                charge_edges = torch.LongTensor(edge_index).to(data.natoms.device)
                charge_edges[0] += torch.sum(data.natoms[:i])
                #charge_edges[1] += torch.sum(data.natoms[:i])
                
                probe_subgraph = data[i]
                
                probe_subgraph.atomic_numbers = torch.cat((probe_subgraph.atomic_numbers, torch.zeros(next_size, device=data.atomic_numbers.device)))
                
                probe_subgraph.pos = torch.cat((probe_subgraph.pos, torch.tensor(charge_pos, device=data.atomic_numbers.device)))
                probe_subgraph.edge_index = charge_edges.long()
                probe_subgraph.edge_distances = edge_distances.to(data.atomic_numbers.device)
                probe_subgraph.cell_offsets = cell_offsets.long().to(data.atomic_numbers.device)
                probe_subgraph.neighbors = torch.tensor([charge_edges.shape[1]]).long().to(data.atomic_numbers.device)
                
                offset += len(charge_pos)
                
                probe_subgraph = data_list_collater([probe_subgraph])
                probe_subgraph.atom_representations = data.atom_representations
                probe_representations.append(self.probe_message_model(probe_subgraph))
        
        probe_results = self.probe_output_function(torch.tensor(probe_representations)).flatten()
        probe_results = torch.reshape(probe_results, data.charge_density.shape)
        
        return probe_results
        '''
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @conditional_grad(torch.enable_grad())
    def forward_atomic(self, data):
        atom_representations = self.atom_message_model(data)
        
        if self.reduce_atom_representations:
            atom_representations = self.atom_reduction(atom_representations)
            
        return(atom_representations)
            
    @conditional_grad(torch.enable_grad())        
    def forward_probe(self, data, atom_representations):
        probe_results = self.probe_message_model(data, atom_representations)
        #probe_results = self.probe_output_function(torch.tensor(probe_representations)).flatten()
        
        return probe_results
        
        
@registry.register_model("dummy_probe")
class dummy(torch.nn.Module):
    def __init__(
        self,
        name = 'Dummy',
        atomic = False,
        probe = True,
        **kwargs,
    ):
        super().__init__()
        self.b = torch.nn.parameter.Parameter(torch.Tensor([1]))
        
    def forward(self, data, atom_representations):
        out = torch.ones(data.input_dict['probe_target'].shape).to(data.input_dict['probe_target'].device)
        out = torch.mul(out, self.b)
        return out