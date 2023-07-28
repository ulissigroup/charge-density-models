import numpy as np
import torch
from pathlib import Path
import ase
import bisect

from tqdm.notebook import tqdm
from multiprocessing import Pool

from ocpmodels.common.registry import registry
from ocpmodels.datasets.ase_datasets import AseDBDataset

from cdm.utils.preprocessing import VaspChargeDensity

@registry.register_dataset('charge_db')
class ChgDBDataset(AseDBDataset):
    '''
    '''
    def __getitem__(self, idx):
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.ids)))]

        # Get atoms object via derived class method
        atoms = self.get_atoms_object(self.ids[idx])

        # Transform atoms object
        if self.atoms_transform is not None:
            atoms = self.atoms_transform(
                atoms, **self.config.get("atoms_transform_args", {})
            )

        if "sid" in atoms.info:
            sid = atoms.info["sid"]
        else:
            sid = torch.tensor([idx])

        # Convert to data object
        data_object = self.a2g.convert(atoms, sid)

        data_object.pbc = torch.tensor(atoms.pbc)
        data_object.charge_density = atoms.info["charge_density"]

        # Transform data object
        if self.transform is not None:
            data_object = self.transform(
                data_object, **self.config.get("transform_args", {})
            )

        return data_object
    
    def get_atoms_object(self, idx):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._idlen_cumulative, idx)

        # Extract index of element within that db
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._idlen_cumulative[db_idx - 1]
        assert el_idx >= 0

        atoms_row = self.dbs[db_idx]._get_row(self.db_ids[db_idx][el_idx])
        atoms = atoms_row.toatoms(add_additional_information = True)

        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)

        return atoms
    
def path_to_list(path):
    path = Path(path)
    return sorted(path.glob("*/CHGCAR"))
    
def write_charge_db(
    CHGCARs,
    ase_db,
    num_workers = 1,
):
    if not isinstance(CHGCARs, list):
        path = Path(CHGCARs)
        ids = sorted(path.glob("*/CHGCAR"))
    else:
        ids = CHGCARs
    
    
    if num_workers == 1:
        for idx in tqdm(ids):
            try:
                vcd = VaspChargeDensity(idx)
                atoms = vcd.atoms[-1]
                dens = vcd.chg[-1]
            except:
                print("Exception occured for: ", idx)

            ase_db.write(atoms, data = {'charge_density':dens})
            
    else:
        batch_size = int(len(ids) / num_workers)
        split_ids = []
        for i in range(num_workers-1):
            split_ids.append(ids[i*batch_size: (i+1) * batch_size])
            
        split_ids.append(ids[(num_workers-1)*batch_size:])
        iterable = zip(split_ids, [ase_db] * num_workers)
        
        with Pool(processes = num_workers) as pool:
            out = pool.starmap(
                func = write_charge_db, 
                iterable = iterable,
            )