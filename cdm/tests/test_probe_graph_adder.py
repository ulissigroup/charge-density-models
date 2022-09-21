import torch

from ase.calculators.vasp import VaspChargeDensity

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
    
    return grid_pos

def test_calculate_grid_pos():
    # Base case
    shape = [2, 2, 2]
    cell = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
    assert (calculate_grid_pos(shape, cell) == old_calculate_grid_pos(shape, cell)).all()
    
    # Non-uniform spacing
    shape = [20, 5, 2]
    cell = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
    assert (calculate_grid_pos(shape, cell) == old_calculate_grid_pos(shape, cell)).all()
    
    # Rectangular cell
    shape = [7, 7, 7]
    cell = [[3, 0, 0],
            [0, 4, 0],
            [0, 0, 5]]
    assert (calculate_grid_pos(shape, cell) == old_calculate_grid_pos(shape, cell)).all()
    
    # Skew cell
    shape = [31, 16, 44]
    cell = [[1, 5, 0],
            [0, 4, 0],
            [0, 1, 6]]
    assert (calculate_grid_pos(shape, cell) == old_calculate_grid_pos(shape, cell)).all()
    
    # Edge case
    shape = [1, 1, 1]
    cell = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
    assert (calculate_grid_pos(shape, cell) == old_calculate_grid_pos(shape, cell)).all()
    
    
def test_get_edges_from_choice():
    vcd = VaspChargeDensity('test_structure')
    
    atoms = vcd.atoms[0]
    dens = vcd.chg[0]
    cell = torch.tensor(np.array([atoms.cell.array]))
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
    
if __name__ == "__main__":
    test_calculate_grid_pos()
    print('Pass: test_calculate_grid_pos')
    
    test_get_edges_from_choice()
    print('Pass: get_edges_from_choice')
    
    