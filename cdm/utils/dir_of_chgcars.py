import numpy as np
import torch

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
   
    '''
    
    def __init__(self, config):
        super(ChgcarDataset, self).__init__()
        self.config = config
        
        self.path = Path(self.config['src'])
        if self.path.is_file():
            raise Exception('The specified src is not a directory')
        split = config.get('split')
        if split is not None:
            f = open(split, "r")
            split = f.readlines()
            self.id = sorted([Path(
                str(self.path) + '/' +
                str(i.rstrip('\n')) + '/CHGCAR' ) for i in split])
        else:
            self.id = sorted(self.path.glob('*/CHGCAR'))
        
        self.a2g = AtomsToGraphs(
            max_neigh = 1000,
            radius = 8,
            r_energy = False,
            r_forces = False,
            r_distances = False,
            r_fixed = False,
            r_pbc = False,
        )
        
        self.transform = config.get('transform')
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        try:
            vcd = VaspChargeDensity(self.id[idx])
            atoms = vcd.atoms[-1]
            dens = vcd.chg[-1]
        except:
            print('Exception occured for: ', self.id[idx])
        
        data_object = self.a2g.convert(atoms)
        data_object.charge_density = dens
        
        if self.transform is not None:
            data_object = self.transform(data_object)
            
        return data_object