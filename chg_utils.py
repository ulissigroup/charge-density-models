import ase.io
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import os
from ocpmodels.preprocessing import AtomsToGraphs
import torch
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data
from ase import Atoms
from ase.calculators.vasp import VaspChargeDensity
import ase.neighborlist as nbl

import pdb
import time

class charge_density:
    def __init__(self, inpath=None, spin_polarized = False):
        self.spin_polarized = spin_polarized

        if self.spin_polarized == True:
            raise NotImplementedError
        
        self.atoms = []
        
        if inpath == None:
            self.cell = [[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]
            self.charge = [[[]]]
            if spin_polarized:
                self.polarization = [[[]]]
        
        elif inpath[-6:] == 'CHGCAR':
            self.read_CHGCAR(inpath)
        
        elif inpath[-5:] == '.cube':
            self.read_cube(inpath)

        else:
            print('Error: Unknown file type. Currently support filetypes are:')
            print('CHGCAR, .cube')
            raise NotImplementedError


    def read_CHGCAR(self, inpath):
        with open(inpath) as CHGCAR:
            lines = CHGCAR.readlines()

            self.name = lines[0][:-1]

            v1, v2, v3 = [lines[i].split() for i in (2, 3, 4)]
            v1 = [float(i) for i in v1]
            v2 = [float(i) for i in v2]
            v3 = [float(i) for i in v3]
            self.cell = [v1, v2, v3]
            self.vol = np.dot(np.cross(v1,v2),v3)

            self.atom_types = lines[5].split()
            atom_counts = lines[6].split()
            self.atom_counts = [int(i) for i in atom_counts]
            self.n_atoms = sum(self.atom_counts)

            k = 0

            for j, element in enumerate(self.atom_types):
                for i in range(self.atom_counts[j]):
                    rel_coords = lines[8+k].split()
                    rel_coords = [float(i) for i in rel_coords]

                    coords = np.array(self.cell).T.dot(rel_coords).tolist()

                    self.atoms.append({'Num': k,
                                    'Name': element,
                                    'pos': coords,
                                    'rel_pos': rel_coords})
                    k += 1

            dims = lines[9+self.n_atoms].split()
            self.grid = [int(i) for i in dims]

            i = 10+self.n_atoms
            chgs = []

            while lines[i].split()[0] != 'augmentation':
                chgs.extend(lines[i].split())
                i += 1

            chgs = [float(x) for x in chgs]

            self.charge = np.reshape(chgs, self.grid)
            self.charge /= self.vol

            for line in lines[i:]:
                tokens = line.split()
                if tokens[0] == 'augmentation':
                    k = int(tokens[-2]) - 1
                    self.atoms[k]['aug'] = []
                else:
                    self.atoms[k]['aug'].extend([float(j) for j in line.split()])


    def write_CHGCAR(self, outpath):
        out = ''
        out += self.name + '\n'
        out += '    1.000000000000000     \n'
        out += f'{self.cell[0][0]:>13.6f}{self.cell[0][1]:>12.6f}{self.cell[0][2]:>12.6f}\n'
        out += f'{self.cell[1][0]:>13.6f}{self.cell[1][1]:>12.6f}{self.cell[1][2]:>12.6f}\n'
        out += f'{self.cell[2][0]:>13.6f}{self.cell[2][1]:>12.6f}{self.cell[2][2]:>12.6f}\n'
        for x in self.atom_types:
            out += f'   {x:<2}'
        out += '\n'
        for x in self.atom_counts:
            out += f'{x:>6}'
        out += '\nDirect\n'

        for atom in self.atoms:
            out += f'  {atom["rel_pos"][0]:.6f}  {atom["rel_pos"][1]:.6f}  {atom["rel_pos"][2]:.6f}\n'

        out += f' \n{self.grid[0]:>5}{self.grid[1]:>5}{self.grid[2]:>5}\n'

        chgs = np.reshape(self.charge, np.prod(self.grid)) * self.vol

        line = ''
        for i, chg in enumerate(chgs):
            line = line + ' '
            if chg >= 1e-12:
                exp = int(np.log10(chg))
                if chg >= 1:
                    exp += 1
                line = line + f'{(chg/10**exp):.11f}' + 'E' + f'{exp:+03}'
            elif chg <= -1e-12:
                exp = int(np.log10(-chg))
                if chg <= -1:
                    exp += 1
                line = line + '-' + f'{(-chg/10**exp):.11f}'[1:] + 'E' + f'{exp:+03}'
            else:
                line = line + '0.00000000000E+00'
            if (i+1) % 5 == 0:
                line = line + '\n'
                out += line
                line = ''

        if line != '':
            out += line + '  \n'

        for k, atom in enumerate(self.atoms):
            line = ''
            out += f'augmentation occupancies{k+1:>4}  '+str(len(atom['aug']))+'\n'
            for i, aug in enumerate(atom['aug']):
                line = line + ' '
                if aug >= 1e-32:
                    exp = int(np.log10(aug))
                    if aug >= 1:
                        exp += 1
                    line = line + ' ' + f'{(aug/10**exp):.7f}' + 'E' + f'{exp:+03}'
                elif aug <= -1e-32:
                    exp = int(np.log10(-aug))
                    if aug <= -1:
                        exp += 1
                    line = line + '-0' + f'{(-aug/10**exp):.7f}'[1:] + 'E' + f'{exp:+03}'
                else:
                    line = line + ' 0.0000000E+00'
                if (i+1) % 5 == 0:
                    line = line + '\n'
                    out += line
                    line = ''
            if line != '':
                out += line + '\n'
        
        with open(outpath, 'w') as file:
            file.write(out)


    def read_cube(self, inpath):
        with open(inpath) as cube:
            lines = cube.readlines()

            self.name = lines[0][:-1]
            self.n_atoms = int(lines[2].split()[0])

            dim1 = int(lines[3].split()[0])
            dim2 = int(lines[4].split()[0])
            dim3 = int(lines[5].split()[0])

            self.grid = [dim1, dim2, dim3]

            v1 = dim1 * np.array([float(x) for x in lines[3].split()[1:]]) * 0.529177 # Converting Bohr to Angstrom
            v2 = dim2 * np.array([float(x) for x in lines[4].split()[1:]]) * 0.529177 # Converting Bohr to Angstrom
            v3 = dim3 * np.array([float(x) for x in lines[5].split()[1:]]) * 0.529177 # Converting Bohr to Angstrom
            
            self.cell = [v1.tolist(), v2.tolist(), v3.tolist()]
            self.vol = np.dot(np.cross(v1, v2), v3)

            element_counts_dict = {}
            self.atoms = []
            
            for i in range(self.n_atoms):
                line = lines[i + 6].split()
                element = int(line[0])
                element = elements_lookup[element - 1]
                p1 = float(line[2]) * 0.529177 # Converting Bohr to Angstrom
                p2 = float(line[3]) * 0.529177 # Converting Bohr to Angstrom
                p3 = float(line[4]) * 0.529177 # Converting Bohr to Angstrom

                coords = [p1, p2, p3]
                rel_coords = np.linalg.inv(np.array(self.cell).T).dot(coords)
                for i, x in enumerate(rel_coords):
                    if x < 0:
                        rel_coords[i] += 1
                
                
                if element in element_counts_dict:
                    element_counts_dict[element] += 1
                else:
                    element_counts_dict[element] = 1
                
                self.atoms.append({'Num': i,
                                   'Name': element,
                                   'pos': coords,
                                   'rel_pos': rel_coords.tolist(),
                                   'aug':[]})

            atom_types = []
            atom_counts = []

            for key, value in element_counts_dict.items():
                atom_types.append(key)
                atom_counts.append(value)
            self.atom_types, self.atom_counts = atom_types, atom_counts
            
            chgs = [float(lines[6 + self.n_atoms + i]) for i in range(np.prod(self.grid))]
            self.charge = np.reshape(chgs, self.grid)

    def write_cube(self, outpath):
        raise NotImplementedError


    def plotly_vis(self):
        raise NotImplementedError


    def __repr__(self):
        out = 'Charge Density Object:\n'
        out += f'Name: {self.name}\n'
        out += f'# of Atoms: {self.n_atoms}\n'
        out += f'Charge Points Grid: {self.grid[0]} {self.grid[1]} {self.grid[2]}\n'
        return out
    

def build_charge_lmdb(inpath, outpath, use_tqdm = False, loud=False):
    a2g = AtomsToGraphs(
        max_neigh = 100,
        radius = 6,
        r_energy = False,
        r_forces = False,
        r_distances = False,
        r_fixed = False,
    )
    
    db = lmdb.open(
        os.path.join(outpath, 'charge.lmdb'),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    
    paths = os.listdir(inpath)
    if use_tqdm:
        paths = tqdm(paths)
        
    for fid, directory in enumerate(paths):
        if loud:
            print(directory)
            
        vcd = VaspChargeDensity(os.path.join(inpath, directory, 'CHGCAR'))
        atoms = vcd.atoms[-1]
        dens = vcd.chg[-1]
        
        #atoms = ase.io.read(os.path.join(inpath, directory, 'OUTCAR'))
        #dens = charge_density(os.path.join(inpath, directory, 'CHGCAR'))

        data_object = a2g.convert(atoms)
        
        data_object.charge_density = dens
        #ata_object.vcd = vcd

        txn = db.begin(write = True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data_object,protocol=-1))
        txn.commit()
    
    
    txn = db.begin(write = True)
    txn.put(f'length'.encode('ascii'), pickle.dumps(fid + 1, protocol=-1))
    txn.commit()
    
    
    db.sync()
    db.close()
    

class ProbeGraphAdder():
    """
    Large portions copied from DeepDFT
    """
    def __init__(self, num_probes=1000, cutoff=5, include_atomic_edges=False):
        self.num_probes = num_probes
        self.cutoff = cutoff
        self.include_atomic_edges = include_atomic_edges
    def __call__(self, data_object):
        probe_data = Data()
        atoms = Atoms(numbers = data_object.atomic_numbers.tolist(),
                      positions = data_object.pos.cpu().detach().numpy(),
                      cell = data_object.cell.cpu().detach().numpy()[0],
                      pbc = [True, True, True])
        
        density = np.array(data_object.charge_density)
        
        grid_pos = _calculate_grid_pos(density, [0,0,0], data_object.cell)
        
        probe_choice_max = np.prod(grid_pos.shape[0:3])
        probe_choice = np.random.randint(probe_choice_max, size = self.num_probes)
        probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
        
        probe_pos = grid_pos[probe_choice[0:3]][:, 0, :]
        probe_target = density[probe_choice]
        
        probe_atoms = Atoms(numbers = [0] * self.num_probes, positions = probe_pos)
        atoms_with_probes = atoms.copy()
        atoms_with_probes.extend(probe_atoms)
        atomic_numbers = atoms_with_probes.get_atomic_numbers()
        
        # probe_data.cell
        probe_data.cell = data_object.cell
        
        # probe_data.atomic_numbers
        probe_data.atomic_numbers = torch.Tensor(atomic_numbers)
            
        # probe_data.natoms
        probe_data.natoms = torch.LongTensor([int(len(atomic_numbers))])
            
        # probe_data.pos
        probe_data.pos = torch.cat((data_object.pos, torch.Tensor(probe_pos)))
        
        # probe_data.target
        probe_data.target = torch.Tensor(probe_target)
        
        # probe_data.edge_index
        probe_edges = []
        probe_offsets = []
        
        neighborlist = AseNeighborListWrapper(self.cutoff, atoms_with_probes)
        
        results = [neighborlist.get_neighbors(i, self.cutoff) for i in range(len(atoms))]
        
        for i, (neigh_idx, neigh_offset) in enumerate(results):
            neigh_atomic_species = atomic_numbers[neigh_idx]
            neigh_is_probe = neigh_atomic_species == 0
            neigh_probes = neigh_idx[neigh_is_probe]
            atom_index = np.ones_like(neigh_probes) * i
            edges = np.stack((atom_index, neigh_probes), axis = 1)
            probe_edges.append(edges)
            probe_offsets.append(neigh_offset[neigh_is_probe])
        
        probe_data.edge_index = torch.tensor(np.concatenate(probe_edges, axis=0)).T
        
        # probe_data.cell_offsets
        probe_data.cell_offsets = torch.tensor(np.concatenate(probe_offsets, axis=0))
        
        # probe_data.neighbors
        
        probe_data.neighbors = torch.LongTensor([probe_data.edge_index.shape[1]])
        
        if self.include_atomic_edges:
            raise NotImplementedError()
        
        data_object.probe_data = probe_data
        return data_object
    
class AseNeighborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist
    Modified from DeepDFT
    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = nbl.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=False
        )
        
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)
        
        return indices, offsets
    
def _calculate_grid_pos(density, origin, cell):
    """
    From DeepDFT
    """
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos