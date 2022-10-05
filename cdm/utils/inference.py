import torch
import warnings
import numpy as np

from tqdm import tqdm

from torch_geometric.data import Batch

from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import data_list_collater

from cdm.chg_utils import ProbeGraphAdder

import pdb

def inference(
    atoms, 
    model, 
    grid = (100, 100, 100), 
    atom_cutoff = 6, 
    probe_cutoff = 6,
    batch_size = 10000, 
    use_tqdm = True, 
    device='cuda'):

    if device is 'cuda' and (torch.cuda.is_available() == False):
        warnings.warn('Cuda not available: running on CPU. Set device="cpu" to avoid this warning')
        device = 'cpu'

    a2g = AtomsToGraphs(
        max_neigh = len(atoms.get_atomic_numbers())**2,
        radius = atom_cutoff,
        r_energy = False,
        r_forces = False,
        r_distances = False,
        r_fixed = False,
    )

    data_object = a2g.convert(atoms)

    data_object.charge_density = torch.zeros(grid)
    data_object.to(device)
    model.to(device)

    pga = ProbeGraphAdder(
        num_probes = batch_size,
        cutoff = probe_cutoff,
        include_atomic_edges = False,
        mode = 'slice',
        slice_start = 0,
        stride = 1,
        implementation = 'RGPBC',
        )

    total_probes = np.prod(grid)
    num_blocks = int(np.ceil(total_probes / batch_size))
    slice_start = 0
    preds = torch.tensor([], device = device)

    loop = range(num_blocks)
    if use_tqdm:
        loop = tqdm(loop)
    
    for i in loop:
        data_object.probe_data = 0
        if i == num_blocks - 1:
            with torch.no_grad():
                data_object = pga(
                    data_object,
                    slice_start = slice_start,
                    num_probes = total_probes - slice_start,
                )
        else:
            with torch.no_grad():
                data_object = pga(
                    data_object,
                    slice_start = slice_start,
                )

        batch = data_list_collater([data_object.clone().detach()])
        batch.probe_data = Batch.from_data_list([data_object.probe_data])

        with torch.no_grad():
            preds = torch.cat((preds, model(batch)))

        slice_start += batch_size
        torch.cuda.empty_cache()

    return torch.reshape(preds, grid)