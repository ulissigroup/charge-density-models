import torch
#from chg_utils import get_probe_graph
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
        atom_model_config = {'name': 'schnet_charge'},
        probe_model_config = {'name': 'schnet_charge'},
        num_interactions = 5,
        atom_channels = 128,
        probe_channels = 128,
        include_atomic_edges = False,
        name = 'charge_model',
    ):
        super().__init__()
        self.regress_forces = False
        
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
        
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # Ensure data has probe points
        assert hasattr(data, 'probe_data')
        
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
        probe_results = self.probe_message_model(data)
        probe_results = self.probe_output_function(probe_results).flatten()
        
        return probe_results
        
'''        
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
'''