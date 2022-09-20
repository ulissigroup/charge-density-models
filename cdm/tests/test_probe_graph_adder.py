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
    
if __name__ == "__main__":
    test_calculate_grid_pos()
    print('Pass: test_calculate_grid_pos')
    