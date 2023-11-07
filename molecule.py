# -*- coding: utf-8 -*-
# @Time    : 6/13/22 8:14 PM
# @Author  : name
# @File    : molecule.py

"""
class 'molecule', 'supramolecule', 'guest'(maybe contain in the class molecule)

function:
    1_Energy_Contrast. rotation with an axis
    2. translation in the binding pocket

Note:
    1_Energy_Contrast. 'potential' has not been implemented, but it is possible to apply on or calculate in class 'Potential'.
"""
import math
import os

# import moldesc
import warnings

from . import pywindow as pw
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from scipy import spatial
from functools import partial

from .atom import Atom
from .bond import Bond
from .cube import Grid
from .utilities import *


class Molecule:
    """
    class 'Molecule'.

    Description of a molecule.

    Functions
    ------

        1_Energy_Contrast. get_atoms() : yield the atoms in the molecule.
        2. get_bonds() : yield the bonds in the molecule.
        3. get_positions() : get the position matrix of the molecule.
        4. get_atom_number() : get the number of atoms in the molecule.
        5. get_centroid() : get the centroid of the molecule, note that it takes H atoms into account.
        6. get_positions_remove_h() : get the position matrix of the molecule without H atoms.
        7. get_centroid_remove_h() : get the centroid of the molecule without H atoms, this function is recommended
                                        to calculate the centroid in the molecule.
        8. convex_hull_on_mol() : get atoms' positions which are on the convex hull in the molecule.
        9. get_farthest_atoms() : get the farthest atoms in the molecule based on function 8.


    """

    def __init__(self, atoms, bonds, positions):
        """
        Initialize a class `Molecule` instance.

        Parameter
        ------
        atoms : `iterable` of `Atom`
            Atoms in the molecule.
        bonds : `iterable` of `Bond`
            Bonds between atoms in the molecule.
        positions : `numpy.ndarray`
            (n , 3) matrix with the position of all atoms in the molecule.

        """
        self._atoms = atoms
        self._bonds = bonds
        self._positions = np.array(positions, dtype=np.float64)

    def translate_a_molecule(self, delta):
        """
        Translate a molecule with a given displacement.

        Parameter
        ------
        delta : `numpy.ndarray`
            Displacement in x, y, z coordinate axis.

        Returns
        ------
        'Molecule'
            New molecule after translation.

        """
        new_positions = translation(self._positions, delta)
        return Molecule(
            self._atoms,
            self._bonds,
            new_positions,
        )

    def get_atoms(self):
        """
        Yield the atoms in the molecule.

        Yields
        ------
        `Atom`
            Atoms in the molecule.

        """
        for atom in self._atoms:
            yield atom

    def get_radii_list(self):
        """
        Return the radii list of the molecule.

        Returns
        -------
        `list`

        """
        radii_list = [atom.get_atomic_radius() for atom in self._atoms]
        return radii_list

    def get_bonds(self):
        """
        Yield the bonds in the molecule.

        Yields
        ------
        `Bond`
            Bonds in the molecule.

        """
        for bond in self._bonds:
            yield bond

    def get_positions(self):
        """
        Get the position matrix of the molecule.

        Returns
        -------
        `numpy.ndarray`
            (n , 3) matrix with the position of all atoms in the molecule.

        """
        return np.array(self._positions, dtype=np.float64)

    def get_atom_number(self):
        """
        Get the number of atoms in the molecule.

        Returns
        -------
        `int`
            Number of atoms in the molecule.

        """
        if len(self._atoms) == 0:
            raise ValueError("No atom in the molecule.")
        return len(self._atoms)

    def get_bond_number(self):
        """
        Get the number of bonds in the molecule.

        Returns
        -------
        `int`
            Number of bonds in the molecule.

        """
        if len(self._bonds) == 0:
            raise ValueError("No bond in the molecule.")
        return len(self._bonds)

    def get_centroid(self):
        """
        Get the centroid of the molecule.

        Returns
        -------
        `numpy.ndarray`
            (3,) vector with the centroid of the molecule.

        """
        if len(self._atoms) != len(self._positions):
            raise ValueError("The number of atoms and positions is not equal.")
        else:
            return np.mean(self._positions, axis=0)

    def get_positions_remove_h(self):
        """
        Get the position matrix of the molecule without H atoms.

        Returns
        -------
        `numpy.ndarray`
            (n , 3) matrix.

        """
        if len(self._atoms) != len(self._positions):
            raise ValueError("The number of atoms and positions is not equal.")
        else:
            positions = self._positions
            H_index = []
            for i in range(len(self._atoms)):
                if self._atoms[i].get_element() == 'H':
                    H_index.append(i)
            positions = np.delete(positions, H_index, axis=0)
            return positions

    def get_centroid_remove_h(self):
        """
        Get the centroid of the molecule without H atoms.

        Returns
        -------
        `numpy.ndarray`
            (3,) vector with the centroid of the molecule.

        """
        positions = self.get_positions_remove_h()
        return np.mean(positions, axis=0)

    def get_info_remove_h(self):
        """
        Get the info of the molecule without H atoms.

        Returns
        -------
        `numpy.ndarray`
            (n , 3) matrix with the position of all atoms in the molecule.

        """
        if len(self._atoms) != len(self._positions):
            raise ValueError("The number of atoms and positions is not equal.")
        else:
            positions = self._positions
            atoms = np.array(self._atoms)
            H_index = []
            for i in range(len(self._atoms)):
                if self._atoms[i].get_element() == 'H':
                    H_index.append(i)
            positions = np.delete(positions, H_index, axis=0)
            atoms = np.delete(atoms, H_index, axis=0)
            return positions, atoms

    def get_farthest_atoms(self):
        """
        Get positions of the farthest atoms to describe the host molecule.

        Returns
        ------
        'numpy.ndarray'
            The positions of the farthest atoms.

        """
        positions = self.get_positions_remove_h()
        # Get involved with scipy.spatial.ConvexHull.
        # Calculate the convex hull in the positions.
        dots_on_convexhull = positions[spatial.ConvexHull(positions, qhull_options='QJ').vertices]

        # Get the distance between dots in convexhull.
        dist_mat = spatial.distance_matrix(dots_on_convexhull, dots_on_convexhull)

        # Get the indices of the farthest atoms.
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

        return dots_on_convexhull[i], dots_on_convexhull[j]

    def convex_hull_on_mol(self):
        """
        Get atoms on the convex hull on molecule.

        Returns
        ------
        'numpy.ndarray'
            The positions of the atoms on convex hull.

        """
        positions = self.get_positions_remove_h()
        # Get involved with scipy.spatial.ConvexHull.
        # Calculate the convex hull in the positions.
        dots_on_convexhull = positions[spatial.ConvexHull(positions, qhull_options='QJ').vertices]

        return dots_on_convexhull

    def _write_to_xyz(self):
        """
        Write Host-Guest complex to .xyz file.

        """
        positions = self.get_positions()
        info = [f"{self.get_atom_number()}\n", f"Host-Guest complex\n"]
        for i, atom in enumerate(self.get_atoms()):
            info.append(
                f'{atom.get_element()} {positions[i][0]} {positions[i][1]} {positions[i][2]}\n'
            )

        return info

    def write_to_xyz(self, output_path, file_name):
        """
        Write Host-Guest complex to .xyz file.

        Parameters
        ------
        output_path : 'str'
            Path to output file.
        file_name : 'str'
            Name of output file.

        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if file_name[-4:] != '.xyz':
            file_name += '.xyz'
        f = open(f'{output_path}/{file_name}', 'w')
        f.writelines(self._write_to_xyz())

    def _write_to_mol(self):
        """
        Write Host-Guest complex to sdf V3000 mol file.

        """
        positions = self.get_positions()
        positions = np.around(positions, 4)
        atom_info = []
        for i, atom in enumerate(self.get_atoms(), 1):
            formal_charge = atom.get_formal_charge()
            if formal_charge != 0:
                formal_charge = f"CHG={atom.get_formal_charge()}"
            else:
                formal_charge = ''

            atom_info.append(
                f"M  V30 {i} {atom.get_element()} {positions[i - 1][0]} "
                f"{positions[i - 1][1]} {positions[i - 1][2]} "
                f"0 {formal_charge}\n"
            )

        bonds = [bond for bond in self.get_bonds()]
        bond_info = []
        for j in range(len(bonds)):
            atom1 = bonds[j].get_atom_1_id() + 1
            atom2 = bonds[j].get_atom_2_id() + 1
            bond_info.append(f"M  V30  {j + 1} "
                             f"{bonds[j].get_bond_type()} "
                             f"{atom1} {atom2}\n"
                             )

        mol_content = [
            '\n',
            '     HostGuestForCages          3D\n',
            '\n',
            '  0  0  0  0  0  0  0  0  0  0999 V3000\n',
            'M  V30 BEGIN CTAB\n',
            f'M  V30 COUNTS {self.get_atom_number()} {len(bonds)} 0 0 0\n',
            'M  V30 BEGIN ATOM\n',
            f"{''.join(atom_info)}",
            'M  V30 END ATOM\n',
            'M  V30 BEGIN BOND\n',
            f"{''.join(bond_info)}",
            'M  V30 END BOND\n',
            'M  V30 END CTAB\n',
            'M  END\n',
            '\n',
            '$$$$\n'
        ]
        return mol_content

    def write_to_mol_file(
            self,
            output_path: str,
            file_name: str,
            convert: bool = True
    ):
        """
        Write 'Molecule' to sdf V3000 mol file.

        Parameters
        ------
        output_path : 'str'
            Path to output file.
        file_name : 'str'
            Name of output file.
        convert : 'bool'
            Convert output file with openbabel, default to be True.

        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if file_name[-4:] != '.mol':
            file_name += '.mol'
        f = open(f'{output_path}/{file_name}', 'w')
        f.writelines(self._write_to_mol())
        f.close()
        if convert:
            os.system(f'obabel "{output_path}/{file_name}" -O "{output_path}/{file_name}" > /dev/null 2>&1')

    def get_disconnected_mol(self):
        """
        Get the disconnected molecule from 'Molecule'.

        Returns
        ------
        """
        graph = nx.Graph()

        # Let atoms be the nodes of the graph.
        for atom in self.get_atoms():
            graph.add_node(atom.get_atom_id())

        # Bonds are the edges of the graph.
        for bond in self.get_bonds():
            e = (bond.get_atom_1_id(), bond.get_atom_2_id())
            graph.add_edge(*e)

        mols = []
        atom_length = []

        for component in nx.connected_components(graph):
            atoms = [
                i for i in self.get_atoms() if i.get_atom_id() in component
            ]

            bonds = [
                i for i in self.get_bonds() if i.get_atom_1_id() in component and i.get_atom_2_id() in component
            ]

            if len(atom_length) == 0:
                atom_length.append(len(atoms))
            else:
                atom_length.append(len(atoms) + atom_length[-1])
                for bond in bonds:
                    bond.reset_atom_1_id(new_atom_1_id=bond.get_atom_1_id() - atom_length[-2])
                    bond.reset_atom_2_id(new_atom_2_id=bond.get_atom_2_id() - atom_length[-2])
            positions = self.get_positions().T[:, list(sorted(component))].T

            mols.append(Molecule(atoms, bonds, positions))
        return tuple(mols)

    def molecule_to_rdkit_mol(self, kekulize=True) -> Chem.Mol:
        """
        Convert 'Molecule' to rdkit 'Mol' object.

        Note
        ---------------
        Please make sure that your 'Molecule' object is kekulized.

        Returns:
            'rdkit.Chem.Mol'

            Test Code
            ---------------


        """
        mol = AllChem.EditableMol(AllChem.Mol())
        for atom in self.get_atoms():
            rdkit_atom = AllChem.Atom(atom.get_atomic_number())
            rdkit_atom.SetFormalCharge(atom.get_formal_charge())
            mol.AddAtom(rdkit_atom)

        for bond in self.get_bonds():
            mol.AddBond(
                beginAtomIdx=bond.get_atom_1_id(),
                endAtomIdx=bond.get_atom_2_id(),
                order=(
                    AllChem.BondType.DATIVE if bond.get_bond_type() == 9
                    else AllChem.BondType(bond.get_bond_type())
                ),
            )

        mol = mol.GetMol()
        rdkit_conf = AllChem.Conformer(self.get_atom_number())
        for atom_id, position in enumerate(self.get_positions()):
            rdkit_conf.SetAtomPosition(atom_id, position)
            mol.GetAtomWithIdx(atom_id).SetNoImplicit(True)
        mol.AddConformer(rdkit_conf)
        Chem.SanitizeMol(mol)
        if kekulize:
            Chem.Kekulize(mol)
        return mol

    def cal_gasteiger_charge(self) -> list:
        """
        Calculate Gasteiger charges for each atom using rdkit.

        Returns
        -------------
        'list'
            List of Gasteiger charges.

        """
        mol = self.molecule_to_rdkit_mol()
        ComputeGasteigerCharges(mol)
        charge = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        return charge

    def search_substructure(self, patt):
        """
        Get the number of patt in the molecule.

        Parameters
        ------
        patt : 'str'
            Substructure to be searched in SMARTS.

        """
        mol = self.molecule_to_rdkit_mol()
        patt = Chem.MolFromSmarts(patt)
        flag = mol.GetSubstructMatches(patt)
        if flag:
            atomids = mol.GetSubstructMatches(patt)
            return atomids, len(atomids)
        else:
            return 'NaN'


class HostMolecule(Molecule):
    """
    class 'HostMolecule'.

    Description of a host molecule such as porous organic cage
    and metal organic cage.

    Init methods
    ------
    1_Energy_Contrast. __init__(atoms, bonds, positions) : init from 'Atom', 'Bond' and 'numpy.ndarray'.

    2. init_from_mol_file(file_path) : init from a molecule file.

    Functions
    ------
    1_Energy_Contrast. rotate_host_molecule(axis, angle) : rotate the host molecule with 'angle' and 'axis'.

    2. translate_to_new_origin(new=np.array([0, 0, 0])) : translate the host molecule to a new centroid.

    3. cal_max_diameter() : calculate the maximum diameter of the host molecule through convex hull algorithm.

    4. cal_pore_diameter() : calculate the diameter of the pore in the host molecule.

    5. cal_pore_volume() : calculate the volume of the pore in the host molecule.

    6. cal_pore_volume_opt() : calculate the optimized volume of the pore in the host molecule.

    7. cal_pore_diameter_opt() : calculate the optimized diameter of the pore in the host molecule.

    8. cal_windows() : calculate windows' size of the host molecule.

    """

    # Input file format associate with rdkit,
    # more format will be considered in the future.
    _input_files = {
        '.mol': partial(AllChem.MolFromMolFile, sanitize=False, removeHs=False),
        '.sdf': partial(AllChem.MolFromMolFile, sanitize=False, removeHs=False)
        # '.mol2': partial(AllChem.MolFromMol2File, sanitize=False, removeHs=False),
    }

    def __init__(self, atoms, bonds, positions):
        """
        Initialize a class `HostMolecule` instance.

        Parameter
        ------
        atoms : `iterable` of `Atom`
            Atoms in the molecule.
        bonds : `iterable` of `Bond`
            Bonds between atoms in the molecule.
        positions : `numpy.ndarray`
            (n , 3) matrix with the position of all atoms in the molecule.

        """
        super(HostMolecule, self).__init__(atoms, bonds, positions)

    def _init_from_rdkit(self, molecule):
        """
        Initialize a class `HostMolecule` instance from an rdkit molecule.

        Parameter
        ------
        molecule : `rdkit molecule`

        """
        atoms = tuple(
            Atom(
                atom_id=atom.GetIdx(), element=atom.GetSymbol(),
                atomic_number=atom.GetAtomicNum(),
                degree=atom.GetDegree(),
                is_aromatic=atom.GetIsAromatic(),
                hybridization=atom.GetHybridization(),
                valence=atom.GetExplicitValence(),
                formal_charge=atom.GetFormalCharge()
            )
            for atom in molecule.GetAtoms()
        )
        bonds = tuple(
            Bond(
                atom_1_id=bond.GetBeginAtomIdx(), atom_2_id=bond.GetEndAtomIdx(),
                bond_id=bond.GetIdx(), bond_type=bond.GetBondType()
            )
            for bond in molecule.GetBonds()
        )

        positions = np.around(np.array(molecule.GetConformer().GetPositions()), 4)

        super().__init__(
            atoms=atoms,
            bonds=bonds,
            positions=positions
        )

    @classmethod
    def init_from_rdkit(cls, molecule):
        """
        Initialize a class `HostMolecule` instance from 'rdkit molecule'.

        Parameter
        ------
        molecule : 'rdkit molecule'

        """
        host_molecule = cls.__new__(cls)
        host_molecule._init_from_rdkit(molecule)
        return host_molecule

    @classmethod
    def init_from_preprocess(
            cls,
            file_path,
            metal,
            from_atoms=(7,8),
            sanitize=True,
            kekulize=True,
            reset_charge=True
    ):
        """
    Preprocessing molecule with wrong bond type and formal charge.

    For example, a metal cage structure which is taken from .cif file, but there's no charge information about the metal
    atoms and the bond type between metal atoms and ligands are all single bond.

    Through this function, you can specify the correct formal charge for the metal atoms.

    Notes
    ------------
    The formal charge is related to atom type assignment in uff4mof force field. If you want to perform this force field
    you should specify the correct formal charge for the metal atoms firstly.

    Parameters
    ------------
    file_path : 'str'
        Path of the molecule file.
    metal: 'dict'
        In the form of {'metal1': new_formal_charge, 'metal2': new_formal_charge, ...}
    from_atoms: 'tuple'
        Replaces bonds between metals and atoms with atomic numbers in from_atoms.
    sanitize: 'bool'
        Whether sanitize the molecule or not. Default to be 'True'.
        If your molecule can not to be kekulized, you should set this parameter to be 'False'.
    kekulize: 'bool'
        Whether to kukulize the molecule or not. Default to be 'True'.
    reset_charge: 'bool'
        Whether reset the formal charge of the molecule or not. Default to be 'True'.
        After correcting the bond type or formal charge, the formal charge for non-metal atoms maybe wrong,
        so you should set this parameter to be 'True' to reset the formal charge = 0.

        Returns
        ------
        'HostMolecule'

        """
        # Read extension name of input file.
        extension_name = os.path.splitext(file_path)[-1]
        if extension_name in cls._input_files.keys():
            molecule = preprocessing(
                    file_path,
                    metal=metal,
                    from_atoms=from_atoms,
                    sanitize=sanitize,
                    kekulize=kekulize,
                    reset_charge=reset_charge
                )
            return cls.init_from_rdkit(molecule)
        else:
            raise ValueError(f"Unsupported file format for {extension_name}!")

    @classmethod
    def init_from_mol_file(
            cls,
            file_path,
            kekulize=True,
            sanitize=False
    ):
        """
        Initialize a class `HostMolecule` instance from a mol file.

        Notes
        ------
        It's not recommended to use 'kekulize=True' when your input molecule
        is hard to read in kekulized format.

        Parameter
        ------
        file_path : 'str'
            Path to the mol file.
        kekulize : 'bool'
            Whether to kekulize the molecule, default to be False.
        sanitize : 'bool'
            Whether to sanitize the molecule, default to be True.


        Returns
        ------
        'HostMolecule'

        """
        # Read extension name of input file.
        extension_name = os.path.splitext(file_path)[-1]
        if extension_name in cls._input_files.keys():
            molecule = reconstruct_mol(cls._input_files[extension_name](file_path))
            # molecule = cls._input_files[extension_name](file_path)
            # Chem.SanitizeMol(molecule)

            # Kekulize molecule with aromatic bond.
            if kekulize:
                AllChem.Kekulize(molecule)
                return cls.init_from_rdkit(
                    molecule
                )
            elif sanitize:
                AllChem.SanitizeMol(molecule)
                return cls.init_from_rdkit(
                    molecule
                )
            else:
                return cls.init_from_rdkit(
                    molecule
                )
        else:
            raise ValueError(f"Unsupported file format for {extension_name}!")

    @classmethod
    def init_from_molecule(cls, molecule):
        """
        Initialize a class `HostMolecule` instance from 'Molecule'.

        Parameter
        ------
        molecule : 'Molecule'

        Returns
        ------
        'HostMolecule'

        """
        host_molecule = cls.__new__(cls)
        host_molecule._atoms = [i for i in molecule.get_atoms()]
        host_molecule._bonds = [i for i in molecule.get_bonds()]
        host_molecule._positions = molecule.get_positions()

        return host_molecule

    def _get_binding_pocket_grid(
            self,
            center_of_cube,
            cube_length,
            number,
            cutoff
    ):
        """
        Get the binding pocket of the host molecule.

                ------Under development.------

        Ongoing
        ------
        In this version, space is calculated by 'cutoff' which is not accurate.
        I will take van der waals radius of each atom into account in the next version.

        Note
        ------
        Binding pocket is defined by the following criteria:
            1_Energy_Contrast. Place a cube on 'center_of_cube' with length 'cube_length';
            2. Split the cube into 'number' grids;
            3. Delete grids which are closer than 'cutoff' with atoms on host molecule;
            4. Delete grids with less than 3 adjacent grids.
            5.  Return the coordinates of remaining grids.

        Parameter
        ------


        Returns
        ------
        'numpy.ndarray'
            (n , 3) matrix with the position of outer grids.

        """
        host_positions = self.get_positions()
        # Get the atom radii.
        atom_radii = np.array([atom.get_atomic_radius() for atom in self.get_atoms()], dtype=np.float)

        # Max and min range of x, y, z coordinates.
        range_limit = cube_length * (number - 1) / (2 * number)
        step_size = cube_length / number

        # Set the range of x, y, z coordinates.
        x_range = [center_of_cube[0] - range_limit, center_of_cube[0] + range_limit]
        y_range = [center_of_cube[1] - range_limit, center_of_cube[1] + range_limit]
        z_range = [center_of_cube[2] - range_limit, center_of_cube[2] + range_limit]

        # Get the coordinates of n**3 grids.
        x_coords = np.arange(x_range[0], x_range[1] + step_size, step_size)
        y_coords = np.arange(y_range[0], y_range[1] + step_size, step_size)
        z_coords = np.arange(z_range[0], z_range[1] + step_size, step_size)
        # print(f'x coordinates : {x_coords}')

        # Get the coordinates of all grids.
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords)
        X.shape, Y.shape, Z.shape = (number ** 3, 1), (number ** 3, 1), (number ** 3, 1)
        grid_coords = np.concatenate((X, Y, Z), axis=1)

        # Create grids in the host molecule.
        all_grids = []
        for i in range(len(grid_coords)):
            # If the grid is on the surface of the cube,
            # 'is_outer' is True.
            if X[i] in x_range or Y[i] in y_range or Z[i] in z_range:
                all_grids.append(Grid(
                    length=step_size,
                    center_position=grid_coords[i],
                    is_outer=True
                ))
            else:
                all_grids.append(Grid(
                    length=step_size,
                    center_position=grid_coords[i],
                    is_outer=False
                ))

        # Scan the distance between atoms and grids along z-axis.
        # If the distance is less than 'radius'+'cutoff', the grid is deleted
        # and 'is_outer' for next grid under the deleted grid is set to True.

        # Traverse all the grids along z-axis.
        for i in range(z_range[0], z_range[1] + step_size, step_size):
            for j in range(y_range[0], y_range[1] + step_size, step_size):
                for k in range(x_range[0], x_range[1] + step_size, step_size):
                    pass

    def rotate_host_molecule(self, axis, angle):
        """
        Rotate a 'HostMolecule' instance with a given axis and angle.

        Parameter
        ------
        axis : 'list'
            The coordinate axis of the rotation.
        angle : `float`
            The angle of the rotation.

        """
        rot_mat = cal_rotation_matrix(axis, angle)
        new_positions = rotation_around_axis(self._positions.T, rot_mat)
        return HostMolecule(
            self._atoms,
            self._bonds,
            new_positions.T,
        )

    def translate_to_new_origin(self, new=np.array([0, 0, 0])):
        """
        Translate original centroid to new original point [0, 0, 0].

        Parameters
        ------
        new : 'numpy.ndarray'
            New centroid to be matched.

        Returns
        ------
        'HostMolecule'
            Host molecule with new original point.

        """
        origin = self.get_centroid_remove_h()
        diff = new - origin
        new_positions = translation(self.get_positions(), diff)
        new_host = HostMolecule(
            atoms=self._atoms,
            bonds=self._bonds,
            positions=np.array(new_positions, dtype=np.float64)
        )

        return new_host

    def _init_pw_instance(self):
        """
        Initialize a 'pywindow.Molecule' instance and
        use 'pywindow' library to calculate information
        about class 'HostMolecule'.

        Notes
        ------
        The function 'load_outer_mol()' is added in 'pywindow/io_tools.Input' and
        'pywindow/molecular.MolecularSystem' to load the molecule instance
        from this code.
            ----------------------------------------
            code added in 'pywindow/io_tools.Input' after function 'load_rdkit_mol(self, mol)':
                # ---Added for use of outer-library information input---
                def load_outer_mol(self, mol):
                    self.system = {
                        'elements': [],
                        'coordinates': np.empty((mol.get_atom_number(), 3))
                }
                for atom in mol.get_atoms():
                    atom_id = atom.get_atom_id()
                    atom_sym = atom.get_element()
                    positions = mol.get_positions()
                    x = positions[atom_id][0]
                    y = positions[atom_id][1_Energy_Contrast]
                    z = positions[atom_id][2]
                    self.system['elements'].append(atom_sym)
                    self.system['coordinates'][atom_id] = x, y, z
                return self.system

            ---------------------------------------------------
            code added in 'pywindow/molecular.MolecularSystem' after function 'load_rdkit_mol(cls, mol)':
                @classmethod
                def load_outer_mol(cls, mol):
                    obj = cls()
                    obj.system = obj._Input.load_outer_mol(mol)
                    return obj

        Returns
        ------
        'pywindow.Molecule'

        References
        ------
        1_Energy_Contrast. J. Chem. Inf. Model. 2018, 58, 12, 2387–2391
        2. https://github.com/marcinmiklitz/pywindow

        """
        molsystem = pw.MolecularSystem.load_outer_mol(self)
        return molsystem.system_to_molecule()

    def cal_max_diameter(self):
        """
        Use convex hull to calculate the maximum diameter of the host molecule.

        Returns
        ------
        'float'
            Maximum diameter of the host molecule.

        """
        atom_on_convex_hull = self.convex_hull_on_mol()

        # Get the distance between dots in convexhull.
        dist_mat = spatial.distance_matrix(atom_on_convex_hull, atom_on_convex_hull)

        # Get the indices of the farthest atoms.
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

        return distance.euclidean(atom_on_convex_hull[i], atom_on_convex_hull[j])

    def cal_pore_diameter(self):
        """
        Use 'pywindow' to calculate the diameter of the pore in host molecule.

        Returns
        ------
        'float'
            The diameter of the pore.

        """
        return self._init_pw_instance().calculate_pore_diameter()

    def cal_pore_volume(self):
        """
        Use 'pywindow' to calculate the volume of the pore in host molecule.

        Returns
        ------
        'float'
            The volume of the pore.

        """
        return self._init_pw_instance().calculate_pore_volume()

    def cal_pore_volume_opt(self):
        """
        Use 'pywindow' to calculate the optimized volume
        of the pore in host molecule.

        Returns
        ------
        'float'
            The optimized volume of the pore.

        """
        return self._init_pw_instance().calculate_pore_volume_opt()

    def cal_pore_diameter_opt(self):
        """
        Use 'pywindow' to calculate the optimized pore diameter
        of the pore in host molecule.

        Returns
        ------
        'float'
            The optimized diameter of the pore.

        """
        return self._init_pw_instance().calculate_pore_diameter_opt()

    def cal_windows(self):
        """
        Use 'pywindow' to calculate windows in host molecule.

        Returns
        ------
        'numpy.ndarray'
            An array of windows' diameters.
        'NoneType'
            If the number of windows is 0.

        """
        mol = self._init_pw_instance()
        return mol.calculate_windows()


class GuestMolecule(Molecule):
    """
    class 'GuestMolecule' for description of guest molecule.

    Attention
    ------
        'GuestMolecule' can be initialized by '.mol', '.sdf' and SMILES.

    Init methods
    ------
    1_Energy_Contrast. __init__(atoms, bonds, positions) : init from 'Atom', 'Bond' and 'numpy.ndarray'.

    2. init_from_mol_file(file_path) : init from a molecule file.

    3. init_from_smiles(smiles) : init from SMILES.

    Functions
    ------
        1_Energy_Contrast. fit_vector_on_molecule() : fit a vector on the molecule through the farthest atoms pair in guest molecule.

        2. translate_to_new_centroid() : translate guest molecule to new centroid.

        3. get_max_length() : get the maximum length of guest molecule.

        4. cal_mol_volume() : calculate volume of guest molecule.

        5. rotate_guest_molecule(axis, angle) : rotate guest molecule with 'axis' and 'angle'.

        6. move_guest_molecule(movement) : move guest molecule with 'movement' vector.

    """
    # Input file format associate with rdkit,
    # more format will be considered in the future.
    _input_files = {
        '.mol': partial(AllChem.MolFromMolFile, sanitize=False, removeHs=False),
        '.sdf': partial(AllChem.MolFromMolFile, sanitize=False, removeHs=False),
    }

    def __init__(self, atoms, bonds, positions):
        """
        Initialize a class `GuestMolecule` instance.

        Parameter
        ------
        atoms : `iterable` of `Atom`
            Atoms in the molecule.
        bonds : `iterable` of `Bond`
            Bonds between atoms in the molecule.
        positions : `numpy.ndarray`
            (n , 3) matrix with the position of all atoms in the molecule.

        """
        super(GuestMolecule, self).__init__(atoms, bonds, positions)

    def _init_from_rdkit(self, molecule):
        """
        Initialize a class `GuestMolecule` instance from an rdkit molecule.

        Parameter
        ------
        molecule : `rdkit molecule`

        """
        atoms = tuple(
            Atom(
                atom_id=atom.GetIdx(), element=atom.GetSymbol(),
                atomic_number=atom.GetAtomicNum(),
                degree=atom.GetDegree(),
                is_aromatic=atom.GetIsAromatic(),
                hybridization=atom.GetHybridization(),
                valence=atom.GetExplicitValence(),
                formal_charge=atom.GetFormalCharge()
            )
            for atom in molecule.GetAtoms()
        )
        bonds = tuple(
            Bond(
                atom_1_id=bond.GetBeginAtomIdx(), atom_2_id=bond.GetEndAtomIdx(),
                bond_id=bond.GetIdx(), bond_type=bond.GetBondType()
            )
            for bond in molecule.GetBonds()
        )
        positions = np.array(molecule.GetConformer().GetPositions())

        super().__init__(
            atoms=atoms,
            bonds=bonds,
            positions=positions
        )

    @classmethod
    def init_from_rdkit(cls, molecule):
        """
        Initialize a class `GuestMolecule` instance from 'rdkit molecule'.

        Parameter
        ------
        molecule : 'rdkit molecule'

        """
        guest_molecule = cls.__new__(cls)
        guest_molecule._init_from_rdkit(molecule)
        return guest_molecule

    @classmethod
    def init_from_mol_file(cls, file_path, kekulize=True, sanitize=False):
        """
        Initialize a class `HostMolecule` instance from a mol file.

        Parameter
        ------
        file_path : 'str'
            Path to the mol file.
        kekulize : 'bool'
            Whether to kekulize the molecule, default to be False.


        Returns
        ------
        'HostMolecule'

        """
        # Read extension name of input file.
        extension_name = os.path.splitext(file_path)[-1]
        if extension_name in cls._input_files.keys():
            # molecule = reconstruct_mol(cls._input_files[extension_name](file_path))
            molecule = cls._input_files[extension_name](file_path)
            # Kekulize molecule with aromatic bond.
            if kekulize:
                AllChem.Kekulize(molecule)
                return cls.init_from_rdkit(
                    molecule
                )
            elif sanitize:
                AllChem.SanitizeMol(molecule)
                return cls.init_from_rdkit(
                    molecule
                )
            else:
                return cls.init_from_rdkit(
                    molecule
                )
        else:
            raise ValueError(f"Unsupported file format for {extension_name}!")

    @classmethod
    def init_from_smiles(cls, smiles: str, AddHs: bool = True, kekulize=True, sanitize=False):
        """
        Initialize a class 'GuestMolecule' instance from SMILES string.

        Parameter
        -------
        smiles : 'str'
            Smiles string of a guest molecule.

        Returns
        ------
        'GuestMolecule'.

        """
        guest = Chem.MolFromSmiles(smiles)

        if AddHs:
            guest = Chem.AddHs(guest)
        if sanitize:
            AllChem.SanitizeMol(guest)
        if kekulize:
            AllChem.Kekulize(guest)
        # Generate conformers of guest molecule.
        conformers = AllChem.EmbedMultipleConfs(guest, numConfs=10, numThreads=0)
        return cls.init_from_rdkit(
            molecule=guest
        )

    @classmethod
    def smiles_to_conformers(cls, num_conformers: int, smiles: str):
        """
        Generate multiple conformers of a molecule through smiles.

        Parameters
        ------
        num_conformers : 'int'
            Number of conformers to generate.
        smiles : 'str'
            SMILES string of the molecule.

        Yields
        ------
        'GuestMolecule'

        """
        guest = AllChem.MolFromSmiles(smiles)
        guest = AllChem.AddHs(guest)
        # Generate multi conformers of guest molecule.
        cids = AllChem.EmbedMultipleConfs(guest, numConfs=num_conformers, randomSeed=1000, numThreads=0)
        try:
            res = AllChem.MMFFOptimizeMoleculeConfs(guest, numThreads=0)
            for i in range(num_conformers):
                conformer = Chem.Mol(guest, True)
                conformer.AddConformer(guest.GetConformer(id=i), assignId=True)
                AllChem.Kekulize(conformer)
                yield cls.init_from_rdkit(
                    molecule=conformer
                )
        except:
            warnings.warn("MMFF optimization failed, yield origin molecule!.")
            yield guest

    @classmethod
    def molfile_to_conformers(cls, num_conformers: int, filepath: str):
        """
        Generate multiple conformers of a molecule through mol file.

        Parameters
        ------
        num_conformers : 'int'
            Number of conformers to generate.
        filepath : 'str'
            Path to the mol file.

        Yields
        ------
        'GuestMolecule'

        """
        guest = AllChem.MolFromMolFile(filepath)
        guest = AllChem.AddHs(guest)
        # Generate multi conformers of guest molecule.
        cids = AllChem.EmbedMultipleConfs(guest, numConfs=num_conformers, randomSeed=1000, numThreads=0)
        try:
            res = AllChem.MMFFOptimizeMoleculeConfs(guest, numThreads=0)
            for i in range(num_conformers):
                conformer = Chem.Mol(guest, True)
                conformer.AddConformer(guest.GetConformer(id=i), assignId=True)
                AllChem.Kekulize(conformer)
                yield cls.init_from_rdkit(
                    molecule=conformer
                )
        except:
            warnings.warn("MMFF optimization failed, yield origin molecule!.")
            yield guest

    @classmethod
    def mol_to_conformers(cls, guest, sel_conformers=20, num_conformers=800, random_seed=1000):
        """
        Generate multiple conformers of a molecule through a 'GuestMolecule' instance.

        Notes
        ------
        If you have a 'GuestMolecule' instance, you can use this method to generate multiple conformers.

        Parameters
        ------
        guest : 'GuestMolecule'
            A 'GuestMolecule' instance.
        num_conformers : 'int'
            Number of conformers to generate.

        Yields
        ------
        'GuestMolecule'

        """
        rdmol = guest.molecule_to_rdkit_mol(kekulize=False)
        rdmol = Chem.AddHs(rdmol)
        cids = AllChem.EmbedMultipleConfs(rdmol, numConfs=num_conformers, randomSeed=random_seed, numThreads=0)
        try:
            ff = AllChem.MMFFGetMoleculeForceField(rdmol, AllChem.MMFFGetMoleculeProperties(rdmol))
            energies = []
            for cid in cids:
                ff.Initialize()
                ff.Minimize(maxIts=200)
                energy = ff.CalcEnergy()
                energies.append(energy)

            # Sort conformers by energy.
            sorted_indices = sorted(range(len(energies)), key=lambda k: energies[k])
            top_indices = sorted_indices[:sel_conformers]
            top_conformers = [rdmol.GetConformer(cid) for cid in [cids[i] for i in top_indices]]
            for i, conf in enumerate(top_conformers):
                conformer = Chem.Mol(rdmol, True)
                conformer.AddConformer(rdmol.GetConformer(id=i), assignId=True)

                # Kekulize the molecule to convert aromatic bonds to the correct double bonds.
                Chem.Kekulize(conformer)

                yield cls.init_from_rdkit(
                    molecule=conformer
                )
        except:
            warnings.warn("MMFF optimization failed, yield origin molecule!.")
            yield guest

    @classmethod
    def chain_to_conformers(cls, guest, sel_conformers=20, num_conformers=800, random_seed=1000):
        pass

    @classmethod
    def init_from_molecule(cls, molecule):
        """
        Initialize a class `GuestMolecule` instance from a molecule.

        Parameters
        ------
        molecule : 'Molecule' instance.

        Returns
        ------
        'GuestMolecule'.

        """
        guest = cls.__new__(cls)
        guest._atoms = [i for i in molecule.get_atoms()]
        guest._bonds = [i for i in molecule.get_bonds()]
        guest._positions = molecule.get_positions()

        return guest

    def fit_vector_on_molecule(self):
        """
        Fit a vector on the guest molecule.

        Returns
        ------
        'numpy.array'
            Vector which is fitted on the guest molecule.

        """
        i, j = self.get_farthest_atoms()
        return np.array(i - j, dtype=np.float)

    def rotation_with_vector(self, origin, end):
        """
        Rotate a molecule from 'origin' vector to 'end' vector.

        Parameters
        ------
        origin : 'numpy.ndarray'
            Origin vector on the molecule.
        end : 'numpy.ndarray'
            End vector to be placed on.

        Returns
        ------
        'GuestMolecule'
            Guest molecule with new position matrix.

        """
        pass

    def translate_to_new_centroid(self, new):
        """
        Translate original centroid to new centroid.
        Such as translate it from [0 0 0] to [1_Energy_Contrast 1_Energy_Contrast 0].

        Parameters
        ------
        new : 'numpy.ndarray'
            New centroid to be matched.

        Returns
        ------
        'GuestMolecule'
            Guest molecule with new position matrix.

        """
        origin = self.get_centroid_remove_h()
        diff = new - origin
        new_positions = translation(self.get_positions(), diff)
        new_guest = GuestMolecule(
            atoms=self._atoms,
            bonds=self._bonds,
            positions=np.array(new_positions, dtype=np.float)
        )

        return new_guest

    def translate_to_new_origin(self, new=np.array([0, 0, 0])):
        """
        Translate original centroid to new original point [0, 0, 0].

        Parameters
        ------
        new : 'numpy.ndarray'
            New centroid to be matched.

        Returns
        ------
        'HostMolecule'
            Host molecule with new original point.

        """
        origin = self.get_centroid_remove_h()
        diff = new - origin
        new_positions = translation(self.get_positions(), diff)
        new_host = GuestMolecule(
            atoms=self._atoms,
            bonds=self._bonds,
            positions=np.array(new_positions, dtype=np.float64)
        )

        return new_host

    def get_max_length(self):
        """
        Get the maximum length of the guest molecule.

        """
        i, j = self.get_farthest_atoms()
        return distance.euclidean(i, j)

    # def cal_mol_volume(self):
    #     """
    #     Using 'MolDesc' code from Craig Waitt to Calculate guest molecule volume.
    #
    #     References
    #     ------
    #     1_Energy_Contrast. https://github.com/cwaitt/MolDesc
    #
    #     """
    #     atom_symbol = [atom.get_element() for atom in self.get_atoms()]
    #     position = self.get_positions()
    #     volume = moldesc.mol_vdw_volume(
    #         mole=ase.Atoms(
    #             atom_symbol,
    #             positions=position
    #         )
    #     )
    #     if volume <= 0:
    #         raise ValueError(f"Impossible molecular volume with : {volume}A**3.")
    #     else:
    #         return volume

    def move_with_vector(self, vector, movement):
        """
        Move a molecule with a vector.

        Parameters
        ------
        vector : 'numpy.ndarray'
            Vector to describe the guest molecule.
        movement : 'numpy.ndarray'
            (1_Energy_Contrast,3) matrix during a movement.

        Returns
        ------
        'GuestMolecule'
            Guest molecule with new position matrix.
        """
        pass

    def rotate_guest_molecule(self, axis, angle):
        """
        Rotate a molecule with axis and angle.

        Parameters
        ------
        axis : 'numpy.ndarray'
            Axis to rotate.
        angle : 'float'
            Angle to rotate in radian.`

        """
        rot_mat = cal_rotation_matrix(np.array(axis), angle)
        new_positions = rotation_around_axis(self._positions.T, rot_mat)
        return GuestMolecule(
            self._atoms,
            self._bonds,
            np.array(new_positions.T, dtype=np.float64)
        )

    def move_guest_molecule(self, movement):
        """
        Move a molecule with a movement.

        Parameters
        ------
        movement : 'numpy.ndarray'
            (1_Energy_Contrast,3) matrix during a movement.

        Returns
        ------
        'GuestMolecule'
            Guest molecule with new position matrix.

        """
        new_positions = translation(self._positions, movement)
        return GuestMolecule(
            self._atoms,
            self._bonds,
            np.array(new_positions)
        )

    def cal_mol_size(self):
        """
        Calculate molecule size.

        Returns
        ------
        'float'
            Minimum and maximum length of the molecule.

        """
        convex_hull_dots = self.convex_hull_on_mol()
        dist_matrix = spatial.distance_matrix(convex_hull_dots, convex_hull_dots)
        none_zero = dist_matrix[np.nonzero(dist_matrix)]
        return np.min(none_zero), np.max(none_zero)


class HostGuestComplex(Molecule):
    """
    class `HostGuestComplex`.

    Representation of a Host-Guest system.

    init methods
    ------
        1_Energy_Contrast. init_from_molecule(host, guest, rotation_axis, rotation_angle, translation_vector) :
            init a 'HostGuestComplex' instance from 'HostMolecule' and 'GuestMolecule' with
            checking of interactions between host atoms and guest atoms.

        2. init_from_molecule_direct(host, guest, rotation_axis, rotation_angle, translation_vector) :
            init a 'HostGuestComplex' instance from 'HostMolecule' and 'GuestMolecule' without
            checking of interactions between host atoms and guest atoms.

        3. init_from_molecule_print_all(host, guest, output_path, rotation_axis, rotation_angle, translation_vector) :
            init a 'HostGuestComplex' instance from 'HostMolecule' and 'GuestMolecule' with
            checking of interactions between host atoms and guest atoms and
            export all the conformers during movement steps.

    Functions
    ------
        1_Energy_Contrast. check_interaction(host_positions, guest_positions, host_atoms, guest_atoms, cutoff=1_Energy_Contrast.15) :
            check if the guest molecule is interacting or overlapping with host molecule.

        2. write_to_xyz(output_path, file_name) : Write 'HostGuestComplex' instance to .xyz file.

    """

    def __init__(self, atoms, bonds, positions, potential=0):
        """
        Initialize a class `HostGuestComplex` instance.

        Parameter
        ------
        atoms : `iterable` of `Atom`
            Atoms in the molecule.
        bonds : `iterable` of `Bond`
            Bonds between atoms in the molecule.
        positions : `numpy.ndarray`
            (n , 3) matrix with the position of all atoms in the molecule.
        potential : 'float'
            Potential energy of host molecule


        """
        super(HostGuestComplex, self).__init__(atoms, bonds, positions)
        self._potential = potential

    @staticmethod
    def check_interaction(
            host_positions,
            guest_positions,
            host_atoms,
            guest_atoms,
            cutoff=1.15
    ):
        """
        Check if the guest molecule is interacting with host molecule.

        Parameters
        ------
        host_positions : 'numpy.ndarray'
            (n , 3) matrix with the position of all atoms in the host molecule.
        guest_positions : 'numpy.ndarray'
            (n , 3) matrix with the position of all atoms in the guest molecule.
        host_atoms : 'iterable' of 'Atom'
            Atoms in the host molecule.
        guest_atoms : 'iterable' of 'Atom'
            Atoms in the guest molecule.
        cutoff : 'float'
            Value to check if the guest molecule is interacting with host molecule.
            Default to be 1_Energy_Contrast.15, refer to 'http://sobereva.com/414'.

        Returns
        ------
            'list'
                A list with atom pairs which are overlapped.

        """
        host_positions = np.array(host_positions)
        guest_positions = np.array(guest_positions)
        host_atoms_radius = [atom.get_covalent_radius() for atom in host_atoms]
        guest_atoms_radius = [atom.get_covalent_radius() for atom in guest_atoms]

        host_guest_distance = distance.cdist(host_positions, guest_positions, 'euclidean')
        interaction = []
        for i in range(len(host_positions)):
            for j in range(len(guest_positions)):
                if host_guest_distance[i][j] < cutoff * (host_atoms_radius[i] + guest_atoms_radius[j]):
                    interaction.append((i, j))
        return interaction

    def _init_from_molecule_test(
            self,
            host,
            guest,
            max_step=10000
    ):
        """
        Note: ------Original TEST Function------
        ------After the development of 'forcefield.potential', this function is not essential
        ------to be used for constructing a 'HostGuestComplex' instance.------
        ------This function will be removed in the future.------
        Initialize a class `HostGuestComplex` instance from 'HostMolecule' and 'GuestMolecule'.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.
        guest : 'GuestMolecule'
            Guest molecule.
        max_step : 'int'
            Maximum number of steps to move the guest molecule, default to be 10000.

        """
        # Translate guest molecule to the centroid of host molecule.
        host = host.translate_to_new_origin()
        new_guest = guest.translate_to_new_centroid([0, 0, 0])

        test_step = 0

        # If len(interaction) != 0, then the guest molecule is interacting with host molecule.
        # Then, the guest molecule will try to translate to new position.
        while True:
            # Calculate information about Host-Guest system.
            host_atoms = [host_atom for host_atom in host.get_atoms()]
            host_bonds = [host_bond for host_bond in host.get_bonds()]
            guest_atoms = [guest_atom for guest_atom in new_guest.get_atoms()]
            guest_bonds = [guest_bond for guest_bond in new_guest.get_bonds()]

            # Reindex the atom id in guest molecule.
            new_guest_atoms = tuple(
                Atom(
                    atom_id=atom.get_atom_id() + len(host_atoms),
                    element=atom.get_element(),
                    atomic_number=atom.get_atomic_number(),
                    degree=atom.get_degree(),
                    is_aromatic=atom.get_is_aromatic(),
                    hybridization=atom.get_hybridization(),
                    valence=atom.get_valence(),
                    formal_charge=atom.get_formal_charge()
                )
                for atom in new_guest.get_atoms()
            )
            new_guest_bonds = tuple(
                Bond(
                    atom_1_id=bond.get_atom_1_id() + len(host_atoms),
                    atom_2_id=bond.get_atom_2_id() + len(host_atoms),
                    bond_id=bond.get_bond_id() + len(host_bonds),
                    bond_type=bond.get_bond_type()
                )
                for bond in new_guest.get_bonds()
            )

            # Check if the guest molecule is interacting with host molecule.
            interaction = self.check_interaction(
                host.get_positions(),
                new_guest.get_positions(),
                host_atoms,
                guest_atoms
            )

            if test_step > max_step:
                raise RuntimeError(f"Failed to find a valid position for guest molecule after {test_step} steps.")

            elif len(interaction) == 0:
                positions = np.concatenate((host.get_positions(), new_guest.get_positions()), axis=0)
                host_atoms.extend(new_guest_atoms)
                host_bonds.extend(new_guest_bonds)
                super().__init__(
                    atoms=tuple(host_atoms),
                    bonds=tuple(host_bonds),
                    positions=positions
                )
                break

            else:
                # Calculate vectors from host to guest in 'interaction'.
                move_vectors = []
                for i in range(len(interaction)):
                    move_vectors.append(
                        host.get_positions()[interaction[i][0]] - new_guest.get_positions()[interaction[i][1]])
                move_vector = -np.sum(np.array(move_vectors), axis=0)
                new_guest = new_guest.move_guest_molecule(move_vector)
                test_step += 1

    @classmethod
    def init_from_molecule_test(
            cls,
            host,
            guest
    ):
        """
        Note: ------Original TEST Function------
        ------After the development of 'forcefield.potential', this function is not essential
        ------to be used for constructing a 'HostGuestComplex' instance.------
        ------This function will be removed in the future.------
        Initialize a class `HostGuestComplex` instance from 'HostMolecule' and 'GuestMolecule'.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.
        guest : 'GuestMolecule'
            Guest molecule.

        """
        host_guest_complex = cls.__new__(cls)
        host_guest_complex._init_from_molecule(host, guest)
        return host_guest_complex

    def _init_from_molecule(
            self,
            host,
            guest,
    ):
        """
        Initialize a class `HostGuestComplex` instance from 'HostMolecule' and 'GuestMolecule'.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.
        guest : 'GuestMolecule'
            Guest molecule.

        """
        # Translate guest molecule to the centroid of host molecule.
        # host = host.translate_to_new_origin()
        # new_guest = guest.translate_to_new_centroid([0, 0, 0])

        # Calculate information about Host-Guest system.
        host_atoms = [host_atom for host_atom in host.get_atoms()]
        host_bonds = [host_bond for host_bond in host.get_bonds()]

        # Reindex the atom id in guest molecule.
        new_guest_atoms = tuple(
            Atom(
                atom_id=atom.get_atom_id() + len(host_atoms),
                element=atom.get_element(),
                atomic_number=atom.get_atomic_number(),
                degree=atom.get_degree(),
                is_aromatic=atom.get_is_aromatic(),
                hybridization=atom.get_hybridization(),
                valence=atom.get_valence(),
                formal_charge=atom.get_formal_charge()
            )
            for atom in guest.get_atoms()
        )
        new_guest_bonds = tuple(
            Bond(
                atom_1_id=bond.get_atom_1_id() + len(host_atoms),
                atom_2_id=bond.get_atom_2_id() + len(host_atoms),
                bond_id=bond.get_bond_id() + len(host_bonds),
                bond_type=bond.get_bond_type()
            )
            for bond in guest.get_bonds()
        )

        positions = np.concatenate((host.get_positions(), guest.get_positions()), axis=0)
        host_atoms.extend(new_guest_atoms)
        host_bonds.extend(new_guest_bonds)

        super().__init__(
            atoms=tuple(host_atoms),
            bonds=tuple(host_bonds),
            positions=positions
        )

    @classmethod
    def init_from_molecule(
            cls,
            host: HostMolecule,
            guest: GuestMolecule,
    ) -> 'HostGuestComplex':
        """
        Initialize a class `HostGuestComplex` instance from 'HostMolecule' and 'GuestMolecule'.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.
        guest : 'GuestMolecule'
            Guest molecule.

        """
        host_guest_complex = cls.__new__(cls)
        host_guest_complex._init_from_molecule(host, guest)

        return host_guest_complex

    def _init_from_multi_molecule(
            self,
            host: HostMolecule = None,
            translate: bool = True,
            guest: tuple[GuestMolecule] = (None,)
    ):
        """
        Initialize a class `HostGuestComplex` instance from 'HostMolecule' and 'GuestMolecule'.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.

        translate : 'bool'
            Whether to translate guest molecules to the centroid of host molecule.
            default to be True.
            note : You can also specify the guests' positions directly
                   with function 'guest.translate_to_new_centroid()'.

        *guest : class 'iterable' of 'GuestMolecule'
            Guest molecule.

        """
        # Translate guest molecule to the centroid of host molecule.
        if not isinstance(guest, (list, tuple)):
            raise TypeError(f"Guest is not an iterable object such as 'tuple' or 'list'.\n")
        host = host.translate_to_new_origin()
        if translate:
            new_guest = [i.translate_to_new_centroid([0, 0, 0]) for i in guest]
            guest_atom_number = sum(i.get_atom_number() for i in guest)
        else:
            new_guest = [i for i in guest]

        # Calculate information about Host-Guest system.
        host_atoms = [host_atom for host_atom in host.get_atoms()]
        host_bonds = [host_bond for host_bond in host.get_bonds()]

        # Reindex the atom id in guest molecule.
        # Handle the tuple of 'GuestMolecule'

        # --------OLD VERSION--------
        # new_guest_atoms = tuple(tuple(
        #     Atom(
        #         atom_id=atom.get_atom_id() + len(host_atoms),
        #         element=atom.get_element(),
        #         atomic_number=atom.get_atomic_number(),
        #         valence=atom.get_valence(),
        #         formal_charge=atom.get_formal_charge()
        #     )
        #     for atom in i.get_atoms()
        # ) for i in new_guest)
        # new_guest_bonds = tuple(tuple(
        #     Bond(
        #         atom_1_id=bond.get_atom_1_id() + len(host_atoms),
        #         atom_2_id=bond.get_atom_2_id() + len(host_atoms),
        #         bond_id=bond.get_bond_id() + len(host_bonds),
        #         bond_type=bond.get_bond_type()
        #     )
        #     for bond in i.get_bonds()
        # )
        #     for i in new_guest)

        # --------NEW VERSION--------
        atom_number = [i.get_atom_number() for i in new_guest]
        add_atom_number = [0]
        for i in range(len(atom_number) - 1):
            add_atom_number.append(sum(atom_number[:i + 1]))

        bond_number = [i.get_bond_number() for i in new_guest]
        add_bond_number = [0]
        for i in range(len(bond_number) - 1):
            add_bond_number.append(sum(bond_number[:i + 1]))

        # Reindex the atom ids.
        new_guest_atoms = sum(
            tuple(
                tuple(
                    Atom(
                        atom_id=atom.get_atom_id() + len(host_atoms) + add_atom_number[i],
                        element=atom.get_element(),
                        atomic_number=atom.get_atomic_number(),
                        degree=atom.get_degree(),
                        is_aromatic=atom.get_is_aromatic(),
                        hybridization=atom.get_hybridization(),
                        valence=atom.get_valence(),
                        formal_charge=atom.get_formal_charge()
                    )
                    for atom in new_guest[i].get_atoms()
                ) for i in range(len(new_guest))),
            ()
        )

        # Reindex the bond ids.
        new_guest_bonds = sum(
            tuple(
                tuple(
                    Bond(
                        atom_1_id=bond.get_atom_1_id() + len(host_atoms) + add_atom_number[i],
                        atom_2_id=bond.get_atom_2_id() + len(host_atoms) + add_atom_number[i],
                        bond_id=bond.get_bond_id() + len(host_bonds) + add_bond_number[i],
                        bond_type=bond.get_bond_type()
                    )
                    for bond in new_guest[i].get_bonds()
                )
                for i in range(len(new_guest))),
            ()
        )

        total_positions = host.get_positions()
        for i in range(len(new_guest)):
            total_positions = np.concatenate((total_positions, new_guest[i].get_positions()), axis=0)
        host_bonds.extend(new_guest_bonds)
        host_atoms.extend(new_guest_atoms)

        super().__init__(
            atoms=tuple(host_atoms),
            bonds=tuple(host_bonds),
            positions=total_positions
        )

    @classmethod
    def init_from_multi_molecule(
            cls,
            host: HostMolecule,
            translate: bool = True,
            *guest
    ) -> 'HostGuestComplex':
        """
        Initialize a class `HostGuestComplex` instance from 'HostMolecule' and 'GuestMolecule'.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.

        translate : 'bool'
            Whether to translate guest molecules to the centroid of host molecule.
            default to be True.
            note : You can also specify the guests' positions directly
                     with function 'guest.translate_to_new_centroid()'.

        *guest : class 'iterable' of 'GuestMolecule'
            Guest molecules.

        """
        host_guest_complex = cls.__new__(cls)
        host_guest_complex._init_from_multi_molecule(
            host,
            translate,
            *guest)
        return host_guest_complex

    def _init_from_molecule_print_all(
            self,
            host,
            guest,
            output_path,
            filename,
            max_step=10000
    ):
        """
        ------Note: This function is used for testing.------
        Print all conformers during optimization.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.
        guest : 'GuestMolecule'
            Guest molecule.
        output_path : 'str'
            Path to output file.
        filename : 'str'
            Name of output file.
        max_step : 'int'
            Maximum number of steps to move the guest molecule, default to be 10000.

        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Translate guest molecule to the centroid of host molecule.
        host = host.translate_to_new_origin()
        new_guest = guest.translate_to_new_centroid(host.get_centroid_remove_h())

        host_atoms = [host_atom for host_atom in host.get_atoms()]
        host_bonds = [host_bond for host_bond in host.get_bonds()]
        guest_atoms = [guest_atom for guest_atom in new_guest.get_atoms()]
        guest_bonds = [guest_bond for guest_bond in new_guest.get_bonds()]

        positions = np.concatenate((host.get_positions(), new_guest.get_positions()), axis=0)
        host_atoms.extend(guest_atoms)
        element = [i.get_element() for i in host_atoms]
        host_bonds.extend(guest_bonds)

        self.write_to_xyz_temp(position_matrix=positions,
                               element_string=element,
                               output_path=output_path,
                               file_name=f"origin_{filename}")

        test_step = 0

        while True:
            # Calculate information about Host-Guest system.
            host_atoms = [host_atom for host_atom in host.get_atoms()]
            host_bonds = [host_bond for host_bond in host.get_bonds()]
            guest_atoms = [guest_atom for guest_atom in new_guest.get_atoms()]
            guest_bonds = [guest_bond for guest_bond in new_guest.get_bonds()]

            # Check if the guest molecule is interacting with host molecule.
            interaction = self.check_interaction(
                host.get_positions(),
                new_guest.get_positions(),
                host_atoms,
                guest_atoms
            )

            print(f'------loop {test_step + 1}------')
            print(f"guest centroid {new_guest.get_centroid_remove_h()}")
            print(f"host centroid {host.get_centroid_remove_h()}")
            print(f"interaction {interaction}")
            print('-------')

            if test_step > max_step:
                raise RuntimeError(f"Failed to find a valid position for guest molecule after {test_step} steps.")

            # If len(interaction) != 0, then the guest molecule is interacting with host molecule.
            # Then, the guest molecule will try to translate to new position.
            elif len(interaction) == 0:
                positions = np.concatenate((host.get_positions(), new_guest.get_positions()), axis=0)
                host_atoms.extend(guest_atoms)
                element = [i.get_element() for i in host_atoms]
                host_bonds.extend(guest_bonds)
                super().__init__(
                    atoms=tuple(host_atoms),
                    bonds=tuple(host_bonds),
                    positions=positions
                )
                break

            else:
                # Calculate vectors from host to guest in 'interaction'.
                move_vectors = []
                for i in range(len(interaction)):
                    move_vectors.append(
                        host.get_positions()[interaction[i][0]] - new_guest.get_positions()[interaction[i][1]])
                move_vector = -np.sum(np.array(move_vectors), axis=0)
                new_guest = new_guest.move_guest_molecule(move_vector)
                positions = np.concatenate((host.get_positions(), new_guest.get_positions()), axis=0)

                host_atoms = [host_atom for host_atom in host.get_atoms()]
                host_bonds = [host_bond for host_bond in host.get_bonds()]
                guest_atoms = [guest_atom for guest_atom in new_guest.get_atoms()]
                guest_bonds = [guest_bond for guest_bond in new_guest.get_bonds()]
                host_atoms.extend(guest_atoms)

                element = [i.get_element() for i in host_atoms]
                host_bonds.extend(guest_bonds)

                self.write_to_xyz_temp(position_matrix=positions,
                                       element_string=element,
                                       output_path=output_path,
                                       file_name=f"{test_step + 1}_{filename}")

                test_step += 1

    @classmethod
    def init_from_molecule_print_all(
            cls,
            host,
            guest,
            output_path,
            filename,
    ):
        """
        ------Note: This function is used for testing.------
        Print all conformers during optimization with checking the interaction
        between 'HostMolecule' and 'GuestMolecule'.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.
        guest : 'GuestMolecule'
            Guest molecule.
        output_path : 'str'
            Path to output file.
        filename : 'str'
            Name of output file.
        """
        host_guest_complex = cls.__new__(cls)
        host_guest_complex._init_from_molecule_print_all(host, guest,
                                                         output_path, filename)
        return host_guest_complex

    def _init_from_molecule_print_all_v2(
            self,
            host,
            guest,
            output_path,
            filename,
            max_step=10000
    ):
        """
        ------Note: This function is used for testing.------
        Print all conformers during optimization.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.
        guest : 'GuestMolecule'
            Guest molecule.
        output_path : 'str'
            Path to output file.
        filename : 'str'
            Name of output file.
        max_step : 'int'
            Maximum number of steps to move the guest molecule, default to be 10000.

        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Translate guest molecule to the centroid of host molecule.
        host = host.translate_to_new_origin()
        new_guest = guest.translate_to_new_centroid(host.get_centroid_remove_h())

        # Calculate information about host and guest molecule.
        # Because these properties are static, we can calculate them before the loop.
        host_atoms = [host_atom for host_atom in host.get_atoms()]
        guest_atoms = [guest_atom for guest_atom in new_guest.get_atoms()]
        host_bonds = [host_bond for host_bond in host.get_bonds()]
        guest_bonds = [guest_bond for guest_bond in new_guest.get_bonds()]

        positions = np.concatenate((host.get_positions(), new_guest.get_positions()), axis=0)

        host_atoms.extend(guest_atoms)
        element = [atom.get_element() for atom in host_atoms]

        self.write_to_xyz_temp(position_matrix=positions,
                               element_string=element,
                               output_path=output_path,
                               file_name=f"origin_struct_{filename}")

        test_step = 0

        while True:
            # Check if the guest molecule is interacting with host molecule.
            interaction = self.check_interaction(
                host.get_positions(),
                new_guest.get_positions(),
                host_atoms,
                guest_atoms
            )

            print(f'------loop {test_step + 1}------')
            print(f"guest centroid {new_guest.get_centroid_remove_h()}")
            print(f"host centroid {host.get_centroid_remove_h()}")
            print(f"interaction {interaction}")
            print('-------')

            if test_step > max_step:
                raise RuntimeError(f"Failed to find a valid position for guest molecule after {test_step} steps.")

            elif len(interaction) == 0:
                positions = np.concatenate((host.get_positions(), new_guest.get_positions()), axis=0)
                host_atoms.extend(guest_atoms)
                host_bonds.extend(guest_bonds)

                super().__init__(
                    atoms=tuple(host_atoms),
                    bonds=tuple(host_bonds),
                    positions=positions
                )
                break

            # If len(interaction) != 0, then the guest molecule is interacting with host molecule.
            # The guest molecule will try to translate to new position.
            else:
                # Calculate vectors from host to guest in 'interaction'.
                move_vectors = []
                old_len = len(interaction)
                for i in range(len(interaction)):
                    move_vectors.append(
                        host.get_positions()[interaction[i][0]] - new_guest.get_positions()[interaction[i][1]])
                move_vector = -np.sum(np.array(move_vectors), axis=0)
                moved_new_guest = new_guest.move_guest_molecule(move_vector)

                # Compare length of interaction with previous step.
                new_interaction = self.check_interaction(
                    host.get_positions(),
                    moved_new_guest.get_positions(),
                    host_atoms,
                    guest_atoms
                )

                # If interaction increased, the movement is invalid and tries to rotate.
                if len(new_interaction) > old_len:
                    pass

                self.write_to_xyz_temp(position_matrix=positions,
                                       element_string=element,
                                       output_path=output_path,
                                       file_name=f"{test_step + 1}_{filename}")

            test_step += 1

    @classmethod
    def init_from_molecule_print_all_v2(
            cls,
            host,
            guest,
            output_path,
            filename
    ):
        """
        ------Note: This function is used for testing.------
        Print all conformers during optimization with checking the interaction
        between 'HostMolecule' and 'GuestMolecule'.

        Parameters
        ------
        host : 'HostMolecule'
            Host molecule.
        guest : 'GuestMolecule'
            Guest molecule.
        output_path : 'str'
            Path to output file.
        filename : 'str'
            Name of output file.
        """
        host_guest_complex = cls.__new__(cls)
        host_guest_complex._init_from_molecule_print_all_v2(host, guest,
                                                            output_path, filename)
        return host_guest_complex

    def _init_target_metal_atom(
            self,
            host,
            guest,
            rotation_axis=np.array([1, 1, 1]),
            rotation_angle=math.pi * 2,
            translation_vector=np.array([0, 0, 0]),
            radius_cutoff=3
    ):
        """
        Initialize a class 'HostGuestComplex' instance with guest target at the metal atom.

        """
        pass

    @staticmethod
    def write_to_xyz_temp(
            position_matrix,
            element_string,
            output_path,
            file_name
    ):
        """
        ------Note: This function is used for testing.------

        """
        position_matrix = np.array(position_matrix, dtype=np.float)
        info = [f"{len(position_matrix)}\n", f"Host-Guest complex\n"]
        for i in range(len(position_matrix)):
            info.append(
                f'{element_string[i]} {position_matrix[i][0]} {position_matrix[i][1]} {position_matrix[i][2]}\n'
            )

        if file_name[-4:] != '.xyz':
            file_name[-4:] = '.xyz'
        f = open(f'{output_path}/{file_name}', 'w')
        f.writelines(info)

    def translate_to_new_origin(self, new=np.array([0, 0, 0])):
        """
        Translate original centroid to new original point [0, 0, 0].

        Parameters
        ------
        new : 'numpy.ndarray'
            New centroid to be matched.

        Returns
        ------
        'HostGuestComplex'
            HostGuestComplex with new original point.

        """
        origin = self.get_centroid_remove_h()
        diff = new - origin
        new_positions = translation(self.get_positions(), diff)
        new_complex = HostGuestComplex(
            atoms=self._atoms,
            bonds=self._bonds,
            positions=np.array(new_positions, dtype=np.float)
        )

        return new_complex
