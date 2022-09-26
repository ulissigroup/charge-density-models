import torch
import os

from ase.build import molecule
from ase.calculators.vasp import VaspChargeDensity

from ocpmodels.common.utils import radius_graph_pbc
from ocpmodels.models.base import BaseModel
from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs
from ocpmodels.datasets import data_list_collater

from cdm.chg_utils import *

def old_calculate_grid_pos(shape, cell):
    ngridpts = np.array(shape) 
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / shape[0],
        np.arange(ngridpts[1]) / shape[1],
        np.arange(ngridpts[2]) / shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    
    return torch.tensor(grid_pos, dtype=torch.float)

def test_calculate_grid_pos():
    # Base case
    shape = [2, 2, 2]
    cell = [[[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]]
    cell = torch.tensor(cell, dtype=torch.float)
    
    assert torch.allclose(calculate_grid_pos(shape, cell), old_calculate_grid_pos(shape, cell))
    
    # Non-uniform spacing
    shape = [20, 5, 2]
    cell = [[[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]]
    cell = torch.tensor(cell, dtype=torch.float)
    
    assert torch.allclose(calculate_grid_pos(shape, cell), old_calculate_grid_pos(shape, cell))
    
    # Rectangular cell
    shape = [7, 7, 7]
    cell = [[[3, 0, 0],
            [0, 4, 0],
            [0, 0, 5]]]
    cell = torch.tensor(cell, dtype=torch.float)
    
    assert torch.allclose(calculate_grid_pos(shape, cell), old_calculate_grid_pos(shape, cell))
    
    # Skew cell
    shape = [31, 16, 44]
    cell = [[[1, 5, 0],
            [0, 4, 0],
            [0, 1, 6]]]
    cell = torch.tensor(cell, dtype=torch.float)
    
    assert torch.allclose(calculate_grid_pos(shape, cell), old_calculate_grid_pos(shape, cell))
    
    # Edge case
    shape = [1, 1, 1]
    cell = [[[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]]
    cell = torch.tensor(cell, dtype=torch.float)
    
    assert torch.allclose(calculate_grid_pos(shape, cell), old_calculate_grid_pos(shape, cell))
    
    
def test_get_edges_from_choice():
    vcd = VaspChargeDensity(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_structure")
    )
    
    atoms = vcd.atoms[0]
    dens = vcd.chg[0]
    cell = torch.tensor(np.array([atoms.cell.array]), dtype=torch.float)
    grid_pos = calculate_grid_pos(dens.shape, cell)
    
    probe_choice = np.random.randint(np.prod(grid_pos.shape[0:3]), size = 100)
    probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
    
    cutoff = 6
    
    out1 = get_edges_from_choice(
        probe_choice, 
        grid_pos,
        atoms, 
        cutoff, 
        include_atomic_edges = False, 
        implementation = 'ASE',
    )
    
    out2 = get_edges_from_choice(
        probe_choice, 
        grid_pos,
        atoms, 
        cutoff, 
        include_atomic_edges = False, 
        implementation = 'RGPBC',
    )
    
    features1 = torch.cat(
        (out1[0], out1[1].T), dim=0
    ).T
    
    features2 = torch.cat(
        (out2[0], out2[1].T), dim=0
    ).T
    
    # Convert rows of tensors to sets. The order of edges is not guaranteed
    features1 = {tuple(x.tolist()) for x in features1}
    features2 = {tuple(x.tolist()) for x in features2}
    
    # Ensure sets are not empty
    assert len(features1) > 0
    assert len(features2) > 0
    
    # Ensure sets are the same
    assert features1 == features2
    
    assert (out1[2] == out2[2]).all()
    assert (out1[3] == out2[3]).all()
    
def test_end_to_end_graph_gen():
    structure = molecule('H')
    structure.cell = [[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10]]
    
    cell = torch.tensor(np.array(structure.cell), dtype=torch.float, device = 'cuda')

    structure.positions = [[0, 9, 0]]
    
    a2g = AtomsToGraphs()
    data = a2g.convert(structure)
    data.charge_density = [[[0]]]
    
    pga = ProbeGraphAdder(num_probes = 1)
    data = pga(data)
    
    model = BaseModel()
    model.otf_graph = False

    (
        edge_index,
        edge_weight,
        distance_vec,
        cell_offsets,
        cell_offset_distances,
        neighbors,
    ) = model.generate_graph(
        data_list_collater([data.probe_data]),
        cutoff = 1000,
        max_neighbors=100,
        use_pbc = True,
        otf_graph = False,
    )
    
    if edge_weight.item() == 19:
        print('Offsets are likely in the wrong direction!')
    
    assert edge_weight.item() == 1
    
if __name__ == "__main__":
    test_calculate_grid_pos()
    print('Pass: test_calculate_grid_pos')
    
    test_get_edges_from_choice()
    print('Pass: test_get_edges_from_choice')
    
    test_end_to_end_graph_gen()
    print('Pass: test_end_to_end_graph_gen')
    
    