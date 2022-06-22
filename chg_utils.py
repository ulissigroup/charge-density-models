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

from DeepDFT.dataset import atoms_and_probe_sample_to_graph_dict, _calculate_grid_pos, collate_list_of_dicts

import pdb

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
    
def get_probe_graph(data, include_atomic_edges = False):
    data.atomic_numbers = torch.cat((data.atomic_numbers, torch.zeros(data.charge_vals.shape, device=data.atomic_numbers.device)))
    data.batch = torch.cat((data.batch, torch.zeros(data.charge_vals.shape, device=data.atomic_numbers.device)))
    
    data.pos = torch.cat((data.pos, data.charge_pos), dim=0)
    
    if include_atomic_edges:
        data.num_atomic_edges = data.edge_representations[0].size()[0]
        data.edge_index = torch.cat((data.edge_index, data.charge_edges), dim=1)
        data.cell_offsets = torch.cat((data.cell_offsets, data.charge_cell_offsets),
                                      dim = 0)
        data.neighbors = data.neighbors + data.charge_neighbors
        
    else:
        data.edge_index = data.charge_edges.T
        data.cell_offsets = data.charge_cell_offsets
        data.neighbors = data.charge_neighbors
                                      
    return data

class AtomsToChargeGraphs(AtomsToGraphs):
    '''
    deprecating
    '''
    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=True,
        r_fixed=False,
        r_charge=True,
        n_pts = 1000,
        mode = 'random',
    ):
        super(AtomsToChargeGraphs, self).__init__(
            max_neigh,
            radius,
            r_energy,
            r_forces,
            r_distances,
            r_edges,
            r_fixed,
        )
        
        self.r_charge = r_charge
        self.n_pts = n_pts
        self.mode = mode
        
    def convert(
        self,
        atoms,
    ):
        """Convert a single atomic stucture to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with edge_index, positions, atomic_numbers,
            and optionally, energy, forces, and distances.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(atoms.get_cell()).view(1, 3, 3)
        natoms = positions.shape[0]

        # put the minimum data in torch geometric data object
        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
        )

        # optionally include other properties
        if self.r_edges:
            # run internal functions to get padded indices and distances
            split_idx_dist = self._get_neighbors_pymatgen(atoms)
            edge_index, edge_distances, cell_offsets = self._reshape_features(
                *split_idx_dist
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.y = energy
        if self.r_forces:
            forces = torch.Tensor(atoms.get_forces(apply_constraint=False))
            data.force = forces
        if self.r_distances and self.r_edges:
            data.distances = edge_distances
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms

                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx

        if self.r_charge:
            struct = AseAtomsAdaptor.get_structure(atoms)
            site_list = []
            
            pts = len(mesh[0]) * len(mesh[1]) * len(mesh[2])
            charge_pos = np.zeros([pts, 3])
            charge_vals = np.zeros(pts)

            point = 0
            for p1 in mesh[0]:
                for p2 in mesh[1]:
                    for p3 in mesh[2]:
                        
                        charge_vals[point] = atoms.charge_density[p1, p2, p3]

                        uc_coords = np.array([p1, p2, p3]) / atoms.charge_grid

                        site = PeriodicSite(
                            species = 1,
                            coords = uc_coords,
                            lattice = struct.lattice,
                            to_unit_cell = True
                        )

                        site_list.append(site)

                        charge_pos[point, :] = site.coords
                        
                        point += 1

                # [to, from] is the convention used by OCP

            a, b, c, d = struct.get_neighbor_list(
                r=self.radius,
                sites = site_list,
                numerical_tol=0, 
                exclude_self=True
            )

            edge_index, edge_distances, cell_offsets = self._reshape_features(
                a, b, d, c
            )

            data.charge_pos = torch.Tensor(charge_pos)
            data.charge_vals = torch.Tensor(charge_vals)
            data.charge_edges = torch.LongTensor(edge_index).flipud()
            data.charge_edges[0] += data.natoms
            data.charge_edges = data.charge_edges.T
            data.charge_cell_offsets = torch.LongTensor(cell_offsets)
            data.charge_neighbors = torch.LongTensor([data.charge_edges.shape[0]])

        return data

def build_charge_lmdb(inpath, outpath, use_tqdm = False, loud=False):
    a2g = AtomsToGraphs(
        max_neigh = 100,
        radius = 6,
        r_energy = True,
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
        atoms = ase.io.read(os.path.join(inpath, directory, 'OUTCAR'))
        dens = charge_density(os.path.join(inpath, directory, 'CHGCAR'))

        data_object = a2g.convert(atoms)
        
        data_object.charge_density = dens.charge

        txn = db.begin(write = True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data_object,protocol=-1))
        txn.commit()
    
    
    txn = db.begin(write = True)
    txn.put(f'length'.encode('ascii'), pickle.dumps(fid + 1, protocol=-1))
    txn.commit()
    
    
    db.sync()
    db.close()
    

class BatchToChargeGraphs(AtomsToGraphs):
    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=True,
        r_fixed=False,
        r_charge=True,
        n_pts = 1000,
        mode = 'random',
    ):
        super().__init__(
            max_neigh,
            radius,
            r_energy,
            r_forces,
            r_distances,
            r_edges,
            r_fixed,
        )

        self.r_charge = r_charge
        self.n_pts = n_pts
        self.mode = mode
    
    def construct_probe_graphs(self, batch):
        # Build atoms objects
        atoms_objects = batch_to_atoms(batch)

        batch.charge_pos = torch.zeros((0,3))
        batch.charge_vals = torch.zeros(0)
        batch.charge_edges = torch.zeros((0,2))
        batch.charge_cell_offsets = torch.zeros((0,3))
        batch.charge_neighbors = torch.zeros(0)
        
        offset = torch.sum(batch.natoms)

        for i, atoms in enumerate(atoms_objects):
            struct = AseAtomsAdaptor.get_structure(atoms)

            # Add probe points
            if self.mode == 'random':
                grid = batch.charge_density[i].shape
                charge_vals = np.zeros(self.n_pts)
                site_list = []
                charge_pos = np.zeros((self.n_pts, 3))

                for point in range(self.n_pts):
                    p1, p2, p3 = [np.random.randint(0, grid[i]) for i in [0, 1, 2]]
                    charge_vals[point] = batch.charge_density[i][p1, p2, p3]
                    uc_coords = np.array([p1, p2, p3]) / grid

                    site = PeriodicSite(
                                species = 1,
                                coords = uc_coords,
                                lattice = struct.lattice,
                                to_unit_cell = True
                            )
                    site_list.append(site)
                    charge_pos[point, :] = site.coords
            
            if self.mode == 'all':
                grid = batch.charge_density[i].shape
                charge_vals = np.zeros(np.prod(grid))
                site_list = []
                charge_pos = np.zeros((np.prod(grid), 3))
                
                point = 0
                for p1 in range(grid[0]):
                    for p2 in range(grid[1]):
                        for p3 in range(grid[2]):
                            charge_vals[point] = batch.charge_density[i][p1, p2, p3]
                            uc_coords = np.array([p1, p2, p3]) / grid
                            
                            site = PeriodicSite(
                                species = 1,
                                coords = uc_coords,
                                lattice = struct.lattice,
                                to_unit_cell = True
                            )
                    site_list.append(site)
                    charge_pos[point, :] = site.coords
                    

            # Get graph information
            a, b, c, d = struct.get_neighbor_list(
            r=self.radius,
            sites = site_list,
            numerical_tol=0, 
            exclude_self=True
        )

            edge_index, edge_distances, cell_offsets = self._reshape_features(
                a, b, d, c
            )

            batch.charge_pos = torch.cat((batch.charge_pos, torch.tensor(charge_pos)))
            batch.charge_vals = torch.cat((batch.charge_vals, torch.tensor(charge_vals)))
            
            charge_edges = torch.LongTensor(edge_index)
            charge_edges[0] += torch.sum(batch.natoms[:i])
            charge_edges[1] += offset
            batch.charge_edges = torch.cat((batch.charge_edges, charge_edges.T))
            
            batch.charge_cell_offsets = torch.cat((batch.charge_cell_offsets, cell_offsets.clone().detach()))
            batch.charge_neighbors = torch.cat((batch.charge_neighbors, torch.tensor([charge_edges.shape[1]])))
            
            offset += len(charge_pos)
            
        #batch.charge_edges = batch.charge_edges.fliplr()
        batch.charge_edges = batch.charge_edges.to(torch.long)
        batch.charge_cell_offsets = batch.charge_cell_offsets.to(torch.long)
        batch.charge_neighbors = batch.charge_neighbors.to(torch.long)

        return batch
    
def batch_to_atoms(batch):
    n_systems = batch.neighbors.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    positions = torch.split(batch.pos, natoms)
    cells = batch.cell

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            cell=cells[idx].cpu().detach().numpy(),
            pbc=[True, True, True],
        )
        atoms_objects.append(atoms)

    return atoms_objects
    
def batch_to_deepDFT_dict(batch, cutoff, num_probes):
    
    list_of_dicts = []
    atoms_objects = batch_to_atoms(batch)
    
    for density, atoms, cell in zip(batch.charge_density, atoms_objects, batch.cell):
        grid_pos = _calculate_grid_pos(density, [0, 0, 0], cell)
        
        input_dict = atoms_and_probe_sample_to_graph_dict(density, atoms, grid_pos, cutoff, num_probes)
        list_of_dicts.append(input_dict)
        
    return collate_list_of_dicts(list_of_dicts)
    
    