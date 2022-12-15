import os
import torch
from ase import Atoms
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from tqdm import tqdm
import warnings

from torch_geometric.data import Batch
from torch_geometric.utils import remove_isolated_nodes, sort_edge_index

from ocpmodels.datasets import data_list_collater
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.common.utils import pyg2_data_transform
from ocpmodels.common.utils import load_state_dict

from cdm.utils.probe_graph import ProbeGraphAdder


@registry.register_model('charge_model')
class ChargeModel(torch.nn.Module):
    def __init__(
        self,
        atom_model_config,
        probe_model_config,
        otf_pga_config = {
            'implementation': 'RGPBC',
        },
        include_atomic_edg es = False,
        enforce_zero_for_disconnected_probes = False,
        enforce_charge_conservation = False,
        freeze_atomic = False,
        name = 'charge_model',
    ):
        super().__init__()
        
        self.regress_forces = False
        self.enforce_zero_for_disconnected_probes = enforce_zero_for_disconnected_probes
        self.enforce_charge_conservation = enforce_charge_conservation
        self.freeze_atomic = freeze_atomic
        
        probe_final_mlp = True
            
        # Initialize atom message-passing model
        if 'checkpoint' in atom_model_config:
            cfg = torch.load(
                atom_model_config['checkpoint'],
                map_location=torch.device('cpu')
            )['config']['model_attributes']
        else:
            cfg = atom_model_config

        
        self.atom_message_model = registry.get_model_class(atom_model_config['name'])(
            **cfg,
            atomic=True, 
            probe=False,
        )
        
        if 'checkpoint' in atom_model_config:
            self.load_checkpoint(
                checkpoint_path = atom_model_config['checkpoint'],
                atomic = True,
            )
        
        
        # Initialize probe message-passing model
        if 'checkpoint' in probe_model_config:
            cfg = torch.load(
                probe_model_config['checkpoint'],
                map_location=torch.device('cpu')
            )['config']['model_attributes']
        else:
            cfg = probe_model_config

        
        self.probe_message_model = registry.get_model_class(probe_model_config['name'])(
            **cfg,
            atomic=False, 
            probe=True,
        )
        
        if 'checkpoint' in probe_model_config:
            self.load_checkpoint(
                checkpoint_path = probe_model_config['checkpoint'],
                probe = True,
            )
        
        # Ensure match between atom and probe messaging models
        if self.atom_message_model.hidden_channels != self.probe_message_model.hidden_channels:
            self.reduce_atom_representations = True
            self.atom_reduction = torch.nn.Sequential(
                torch.nn.Linear(self.atom_message_model.hidden_channels,self.atom_message_model.hidden_channels),
                torch.nn.Sigmoid(),
                torch.nn.Linear(self.atom_message_model.hidden_channels, self.probe_message_model.hidden_channels))
        else:
            self.reduce_atom_representations = False
            
        assert self.atom_message_model.num_interactions >= self.probe_message_model.num_interactions
        
        # Compatibility for specific models
        if probe_model_config['name'] == 'scn_charge':
            probe_final_mlp = False
        
        if probe_final_mlp:
            self.probe_output_function = torch.nn.Sequential(
                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Linear(self.probe_message_model.hidden_channels, 1)
            )
            
        self.otf_pga = ProbeGraphAdder(**otf_pga_config)
        
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # Ensure data has probe points
        if not hasattr(data, 'probe_data'):
            data = self.otf_pga(data)
            data.probe_data = [pyg2_data_transform(data.probe_data)]
            data.probe_data = Batch.from_data_list(data.probe_data)
        
        atom_representations = self.forward_atomic(data)

        probes = self.forward_probe(data.probe_data, atom_representations)
        
        return probes
    
    @conditional_grad(torch.enable_grad())
    def forward_atomic(self, data):
        data.edge_index = sort_edge_index(data.edge_index.flipud()).flipud()
        
        if self.freeze_atomic:
            with torch.no_grad():
                atom_representations = self.atom_message_model(data)
        else:
            atom_representations = self.atom_message_model(data)
        
        if self.reduce_atom_representations:
            atom_representations = [self.atom_reduction(rep).float() for rep in atom_representations]
            
        return(atom_representations)
            
    @conditional_grad(torch.enable_grad())        
    def forward_probe(self, data, atom_representations):
        data.atom_representations = atom_representations[-self.probe_message_model.num_interactions:]
        
        data.edge_index = sort_edge_index(data.edge_index.flipud()).flipud()
        
        probe_results = self.probe_message_model(data)
        
        if hasattr(self, 'probe_output_function'):
            probe_results = self.probe_output_function(probe_results).flatten()
            
        probe_results = torch.nan_to_num(probe_results)
        
        if self.enforce_zero_for_disconnected_probes:
            is_probe = data.atomic_numbers == 0
            _, _, is_not_isolated = remove_isolated_nodes(data.edge_index, num_nodes = len(data.atomic_numbers))
            is_isolated = ~is_not_isolated
            probe_results[is_isolated[is_probe]] = torch.zeros_like(probe_results[is_isolated[is_probe]])

        if self.enforce_charge_conservation: 
            if torch.sum(probe_results) == 0:
                warnings.warn('Charge prediction is 0 - cannot enforce charge conservation!')
            else:
                data.total_target = data.total_target.to(probe_results.device)
                probe_results *= data.total_target / torch.sum(probe_results)
        
        return probe_results
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    
    def load_checkpoint(
        self,
        checkpoint_path,
        atomic=False,
        probe=False
    ):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        if atomic:
            model = self.atom_message_model
        if probe:
            model = self.probe_message_model
        
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]
            
        load_state_dict(model, new_dict, strict=False)