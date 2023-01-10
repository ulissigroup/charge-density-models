import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from ocpmodels.common.registry import registry
from ocpmodels.preprocessing import AtomsToGraphs

from cdm.utils.preprocessing import VaspChargeDensity

@registry.register_dataset('dir_of_chgcars')
class ChgcarDataset(Dataset):
    '''
    This Dataset is used to process the outputs of VASP calculations directly.
    The following directory structure is expected:
    
    src/
    ├─ calculation_1/
       ├─ CHGCAR
       ├─ ...
    ├─ calculation_2/
       ├─ CHGCAR
       ├─ ...
    ├─ ...
    
    config['cutoff'] must be set explicitly.
   
    '''
    
    def __init__(self, config, transform = None):
        super(ChgcarDataset, self).__init__()
        self.config = config
        self.transform = transform
        
        self.path = Path(self.config['src'])
        if self.path.is_file():
            raise Exception('The specified src is not a directory')
            
        self.paths = sorted(self.path.glob('*/CHGCAR'))
        self.a2g = AtomsToGraphs(
            max_neigh = 1000,
            radius = self.config['cutoff'],
            r_energy = False,
            r_forces = False,
            r_distances = False,
            r_fixed = False,
            r_pbc = False,
        )
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        vcd = VaspChargeDensity(self.paths[idx])
        atoms = vcd.atoms[-1]
        dens = vcd.chg[-1]
        
        data_object = self.a2g.convert(atoms)
        data_object.charge_density = dens
        
        if self.transform is not None:
            data_object = self.transform(data_object)
            
        return data_object