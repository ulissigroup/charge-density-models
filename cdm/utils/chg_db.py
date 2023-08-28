import numpy as np
import torch
import ase
import bisect
import warnings

from tqdm.notebook import tqdm
from pathlib import Path
from multiprocessing import Pool

from ocpmodels.common.registry import registry
from ocpmodels.datasets.ase_datasets import AseDBDataset

from cdm.utils.preprocessing import VaspChargeDensity

@registry.register_dataset('charge_db')
class ChgDBDataset(AseDBDataset):
    '''
    An alternative database format based on ASE databases
    One way to create such a database is with the "write_charge_db"
    script in this file. This script requires VASP CHGCARs.
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
        
        if isinstance(data_object.charge_density, list):
            data_object.charge_density = torch.tensor(data_object.charge_density)

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
    
    for idx in tqdm(ids):
        try:
            vcd = VaspChargeDensity(idx)
            atoms = vcd.atoms[-1]
            dens = vcd.chg[-1]
        except:
            print("Exception occured for: ", idx)

        try:
            ase_db.write(atoms, data = {'charge_density': dens})
        except TypeError:
            warnings.warn("Failed to write tensor to database. Trying again as a list!")
            ase_db.write(atoms, data = {'charge_density': dens.tolist()})