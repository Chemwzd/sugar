import rdkit
from rdkit.Chem.rdchem import BondType


class Bond:
    """
    class 'Bond'.

    Description of a bond between two atoms.

    """

    _bond_type = {
        'SINGLE': 1,
        'DOUBLE': 2,
        'TRIPLE': 3,
        'AROMATIC': 4,
        'DATIVE': 9
    }

    def __init__(self, atom_1_id, atom_2_id, bond_id, bond_type: str):
        """
        Initialize a class `Bond` instance.

        Parameters
        ----------
        atom_1_id : `int`
            ID of atom 1 in bond.
        atom_2_id : `int`
            ID of atom 2 in bond.
        bond_id : `int`
            Bond ID in a molecule.
        bond_type : `str`
            Bond type.

        """
        if atom_1_id == atom_2_id:
            raise ValueError('This bond is invalid with same atom id.')
        self._atom_1_id = atom_1_id
        self._atom_2_id = atom_2_id
        self._bond_id = bond_id
        self._bond_type = bond_type

    def get_bond_id(self):
        """
        Get ID of a specified bond.

        returns
        -------
        'int'
            Bond ID of the specified bond.

        """
        return self._bond_id

    def get_atom_1_id(self):
        """
        Get ID of atom 1 in bond.

        returns
        -------
        'int'
            ID of atom 1 in bond.

        """
        return self._atom_1_id

    def get_atom_2_id(self):
        """
        Get ID of atom 2 in bond.

        returns
        -------
        'int'
            ID of atom 2 in bond.

        """
        return self._atom_2_id

    def get_bond_type(self):
        """
        Get bond type.

        returns
        -------
        'float'
            Bond type as a float.

        """
        rd_bond_type = {
            BondType.SINGLE: 1,
            BondType.DOUBLE: 2,
            BondType.TRIPLE: 3,
            BondType.AROMATIC: 4,
            BondType.DATIVE: 9,
        }
        if isinstance(self._bond_type, int):
            return rd_bond_type[self._bond_type]
        if self._bond_type in rd_bond_type:
            return rd_bond_type[self._bond_type]
        elif str(self._bond_type) in self._bond_type:
            return self._bond_type[str(self._bond_type)]
        else:
            raise ValueError(f'Invalid bond type {self._bond_type}.')

    def reset_atom_1_id(self, new_atom_1_id):
        """
        Reset atom 1 id in bond.
        """
        self._atom_1_id = new_atom_1_id

    def reset_atom_2_id(self, new_atom_2_id):
        """
        Reset atom 2 id in bond.
        """
        self._atom_2_id = new_atom_2_id

    def reset_bond_id(self, new_bond_id):
        """
        Reset bond id in bond.
        """
        self._bond_id = new_bond_id

    def reset_bond_type(self, new_bond_type):
        """
        Reset bond type in bond.
        """
        self._bond_type = new_bond_type
