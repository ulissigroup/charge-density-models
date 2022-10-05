import torch
from ase import Atoms
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import pdb
from tqdm import tqdm
from torch_geometric.utils import remove_isolated_nodes

from ocpmodels.common.utils import conditional_grad
from ocpmodels.common.registry import registry
from ocpmodels.datasets import data_list_collater

from cdm.utils.probe_graph import ProbeGraphAdder

@registry.register_model("charge_model")
class ChargeModel(torch.nn.Module):
    def __init__(
        self,
        atom_model_config = {'name': 'schnet_charge'},
        probe_model_config = {'name': 'schnet_charge'},
        otf_pga_config = {'num_probes': 500,
                          'cutoff': 4,
                          'include_atomic_edges': False,
                          'mode': 'random',
                          'stride': 1,
                          'implementation': 'RGPBC',
                         },
        num_interactions = 5,
        atom_channels = 128,
        probe_channels = 128,
        include_atomic_edges = False,
        enforce_zero_for_disconnected_probes = False,
        name = 'charge_model',
    ):
        super().__init__()
        self.regress_forces = False
        self.enforce_zero_for_disconnected_probes = enforce_zero_for_disconnected_probes
        
        self.probe_output_function = torch.nn.Sequential(
            torch.nn.Linear(probe_channels, probe_channels),
            torch.nn.ELU(),
            torch.nn.Linear(probe_channels, 1)
        )

        
        self.atom_message_model = registry.get_model_class(atom_model_config['name'])(**atom_model_config, atomic=True, 
                      probe=False,
                      hidden_channels = atom_channels,
                      num_interactions = num_interactions)
        
        self.probe_message_model = registry.get_model_class(probe_model_config['name'])(**probe_model_config, atomic=False,
                       probe=True,
                       hidden_channels = probe_channels, 
                       num_interactions = num_interactions)
        
        if atom_model_config['name'] == 'gemnet_charge' or probe_model_config['name'] == 'gemnet_charge':
            self.include_atomic_edges = True
        else:
            self.include_atomic_edges = False
            
        if atom_channels != probe_channels:
            self.reduce_atom_representations = True
            self.atom_reduction = torch.nn.Sequential(
                torch.nn.Linear(atom_channels, atom_channels),
                torch.nn.Sigmoid(),
                torch.nn.Linear(atom_channels, probe_channels))
        else:
            self.reduce_atom_representations = False
            
        self.otf_pga = ProbeGraphAdder(
            num_probes = otf_pga_config['num_probes'],
            cutoff = otf_pga_config['cutoff'],
            include_atomic_edges = otf_pga_config['include_atomic_edges'],
            mode = otf_pga_config['mode'],
            stride = otf_pga_config['stride'],
            implementation = otf_pga_config['implementation'],
        )
        
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # Ensure data has probe points
        if not hasattr(data, 'probe_data'):
            data = self.otf_pga(data)
        
        atom_representations = self.forward_atomic(data)

        probes = self.forward_probe(data.probe_data, atom_representations)
        
        return probes
        
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @conditional_grad(torch.enable_grad())
    def forward_atomic(self, data):
        atom_representations = self.atom_message_model(data)
        
        if self.reduce_atom_representations:
            atom_representations = [self.atom_reduction(rep).float() for rep in atom_representations]
            
        return(atom_representations)
            
    @conditional_grad(torch.enable_grad())        
    def forward_probe(self, data, atom_representations):
        
        data.atom_representations = atom_representations
        probe_representations = self.probe_message_model(data)
        probe_results = self.probe_output_function(probe_representations).flatten()
        
        if self.enforce_zero_for_disconnected_probes:
            is_probe = data.atomic_numbers == 0
            _, _, is_not_isolated = remove_isolated_nodes(data.edge_index, num_nodes = len(data.atomic_numbers))
            is_isolated = ~is_not_isolated
            probe_results[is_isolated[is_probe]] = torch.zeros_like(probe_results[is_isolated[is_probe]])
        
        return probe_results