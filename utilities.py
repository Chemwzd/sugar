# -*- coding: utf-8 -*-
# @Time    : 6/13/22 8:23 PM
# @Author  : wzd
# @File    : utilities.py
import numpy as np
import scipy.linalg as linalg
import networkx as nx
import random
import rdkit

from scipy.spatial import distance, distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem

from sugar.atom import _metal_element


def translation(position, move):
    """
    Translate origin position with move_matrix.

    Parameter
    ------
    position : 'numpy.ndarray'
        The position matrix in 3D coordinate.
    move : 'numpy,ndarray'
        Movement vector.


    Returns
    ------
    new_position : 'numpy.ndarray'
        Return position matrix after one step of movement.

    """
    position = np.array(position)
    new_position = np.add(position, np.array(move))
    return new_position


def rotation_matrix(angle, axis_x, axis_y, axis_z):
    """
    Calculate rotation matrix around axis in given angle.

    Parameter
    ------
    angle : 'float'
        Angle during rotation in radian.
    axis_x : 'float'
        X coordinate of the rotation axis.
    axis_y : 'float'
        Y coordinate of the rotation axis.
    axis_z : 'float'
        Z coordinate of the rotation axis.

    Returns
    ------
    'numpy.ndarray'
        Rotation matrix.

    """
    axis = np.array([axis_x, axis_y, axis_z])
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * angle))
    return rot_matrix


def cal_rotation_matrix(axis, angle):
    """
    Calculate rotation matrix around axis in given angle.

    Parameter
    ------
    origin : 'numpy.ndarray'
        Origin position.
    axis : 'numpy.ndarray'
        Rotation axis.
    angle : 'float'
        Angle during rotation in radian.

    """
    axis = np.asarray(axis) / np.linalg.norm(axis)
    # calculate the rotation matrix
    rot_matrix = np.array([
        [axis[0] ** 2 * (1 - np.cos(angle)) + np.cos(angle),
         axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
         axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
        [axis[0] * axis[1] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
         axis[1] ** 2 * (1 - np.cos(angle)) + np.cos(angle),
         axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
        [axis[0] * axis[2] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
         axis[1] * axis[2] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
         axis[2] ** 2 * (1 - np.cos(angle)) + np.cos(angle)],
    ])
    return np.array(rot_matrix).reshape((3, 3))


def rotation_around_axis(position, rot_mat):
    """
    Rotation origin position with rotation matrix.

    Parameter
    ------
    position : 'numpy.ndarray'
        Origin position.
    rota_mat : 'numpy.ndarray'
        Rotation matrix which is applied to rotation operation.

    Returns
    ------
    'numpy.ndarray'
        The new position after rotation.

    """
    return np.dot(rot_mat, position)


def rotation_with_axis(position, axis, angle):
    """
    Rotate origin position with axis and angle.

    Parameters
    ------
    position : 'numpy.ndarray'
        Origin position.

    axis : 'numpy.ndarray'
        Rotation axis.

    angle : 'float'
        Angle during rotation in radian.

    """
    rot_mat = cal_rotation_matrix(np.array(axis), angle)
    return rotation_around_axis(position, rot_mat)


def distance_between_dot(position_1, position_2):
    """
    Calculate the distance between two dots in 3D coordinate.

    Parameter
    -----
    position_1 : 'numpy.ndarray'
        Position matrix of dot1.
    position_2 : 'numpy.ndarray'
        Position matrix of dot2.

    Returns
    ------
    'float'
        Distance between two dots.

    """
    return np.linalg.norm(position_1 - position_2)


def exam_move(energy_old, energy_new, mc_constant=2):
    if energy_new < energy_old:
        return True
    else:
        value = np.exp(mc_constant * (energy_new - energy_old))
        # TEST
        # value = np.exp(-(energy_new - energy_old) / 4)
        random_value = random.random()
        # TEST
        # print(f"value : {value}; random_value: {random_value}\n")
        if value > random_value:
            return True
        else:
            return False


def norm_vector(vector):
    return np.array(vector) / np.linalg.norm(vector)


# @jit
def cal_dis_mat(dot_positions, position_matrix):
    """
    Calculate the distance matrix between all dots and all positions.

    Parameters
    ------
    dot_positions : 'numpy.ndarray'
        Position matrix of all dots.

    position_matrix : 'numpy.ndarray'
        Position matrix of all positions to be calculated.

    Returns
    ------
    'numpy.ndarray'
        Distance matrix.

    """
    dis_mat = distance_matrix(dot_positions, position_matrix)
    return dis_mat


def calc_rmsd(mol1, mol2):
    """
    Calculate the RMSD between two molecules.

    Parameters
    -------------------
    mol1 : 'Molecule'
    mol2 : 'Molecule'

    Returns
    -------------------
    'float'
        RMSD between two molecules in Å.

    """
    mol1 = mol1.translate_to_new_origin([0, 0, 0])
    mol2 = mol2.translate_to_new_origin([0, 0, 0])

    positions1 = mol1.get_positions()
    positions2 = mol2.get_positions()

    deviations = positions1 - positions2
    rmsd = np.sqrt(np.sum(deviations ** 2) / len(positions1))

    return rmsd


def reconstruct_mol(molecule):
    """
    Reconstruct a molecule from scratch.

    Parameters
    ----------
    molecule : `rdkit.Mol`

    Returns
    -------
    `rdkit.Mol`

    """
    emol = AllChem.EditableMol(AllChem.Mol())
    for atom in molecule.GetAtoms():
        new = AllChem.Atom(atom.GetAtomicNum())
        new.SetFormalCharge(atom.GetFormalCharge())
        emol.AddAtom(new)

    for bond in molecule.GetBonds():
        emol.AddBond(
            beginAtomIdx=bond.GetBeginAtomIdx(),
            endAtomIdx=bond.GetEndAtomIdx(),
            order=bond.GetBondType()
        )

    m = emol.GetMol()
    m.AddConformer(molecule.GetConformer())
    return m


def preprocessing(
        file_path,
        metal=None,
        from_atoms=(7, 8),
        sanitize=False,
        kekulize=True,
        reset_charge=True
):
    """
    Preprocessing molecule with wrong bond type and formal charge.

    For example, a metal cage structure which is taken from .cif file, but there's no charge information about the metal
    atoms and the bond type between metal atoms and ligands are all single bond.
    Through this function, you can specify the correct formal charge for the metal atoms.

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

    Adapt from rdkit Cookbook:
            https://github.com/rdkit/rdkit/blob/8f4d4a624c69df297552cabb2d912b2ac7c39e84/Docs/Book/Cookbook.rst#L2174
    """
    def is_transition_metal(at):
        n = at.GetAtomicNum()
        return (n >= 22 and n <= 29) or (n >= 40 and n <= 47) or (n >= 72 and n <= 79)

    if metal is None or not isinstance(metal, dict):
        raise ValueError(f"Metal input should be a dict.")
    temp_mol = AllChem.MolFromMolFile(file_path, sanitize=False, removeHs=False)
    # temp_mol = AllChem.AddHs(temp_mol)
    pt = Chem.GetPeriodicTable()

    rw_mol = Chem.RWMol(temp_mol)
    rw_mol.UpdatePropertyCache(strict=False)
    metals = [at for at in rw_mol.GetAtoms() if is_transition_metal(at)]

    for m in metals:
        m.SetFormalCharge(metal[m.GetSymbol()])
        for nbr in m.GetNeighbors():
            if nbr.GetAtomicNum() in from_atoms and \
                    nbr.GetExplicitValence() > pt.GetDefaultValence(nbr.GetAtomicNum()) and \
                    rw_mol.GetBondBetweenAtoms(nbr.GetIdx(), m.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rw_mol.RemoveBond(nbr.GetIdx(), m.GetIdx())
                rw_mol.AddBond(nbr.GetIdx(), m.GetIdx(), Chem.BondType.DATIVE)
    if reset_charge:
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() not in _metal_element:
                atom.SetFormalCharge(0)
    if sanitize:
        AllChem.SanitizeMol(rw_mol)
    if kekulize:
        AllChem.Kekulize(rw_mol)
    return rw_mol


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
