# -*- coding: utf-8 -*-
# @Time    : 6/13/22 8:16 PM
# @Author  : wzd
# @File    : atom.py


"""
1_Energy_Contrast. Van der Waals atomic radii in Å from Santiago Alvarez.
2. CSD Covalent radii in Å from Santiago Alvarez.

Reference
------------
1_Energy_Contrast. Dalton Trans., 2013, 42, 8617–8636
2. Dalton Trans., 2008, 2832-2838

"""
_vdw_radius = {
    'H': 1.09, 'He': 1.43, 'Li': 2.12, 'Be': 1.98, 'B': 1.91, 'C': 1.77,
    'N': 1.66, 'O': 1.50, 'F': 1.46, 'Ne': 1.58, 'Na': 2.50, 'Mg': 2.51,
    'Al': 2.25, 'Si': 2.19, 'P': 1.90, 'S': 1.89, 'Cl': 1.82, 'Ar': 1.83,
    'K': 2.73, 'Ca': 2.62, 'Sc': 2.58, 'Ti': 2.46, 'V': 2.42, 'Cr': 2.45,
    'Mn': 2.45, 'Fe': 2.44, 'Co': 2.40, 'Ni': 2.40, 'Cu': 2.38, 'Zn': 2.39,
    'Ga': 2.32, 'Ge': 2.29, 'As': 1.88, 'Se': 1.82, 'Br': 1.86, 'Kr': 2.25,
    'Rb': 3.21, 'Sr': 2.84, 'Y': 2.75, 'Zr': 2.52, 'Nb': 2.56, 'Mo': 2.45,
    'Tc': 2.44, 'Ru': 2.46, 'Rh': 2.44, 'Pd': 2.15, 'Ag': 2.53, 'Cd': 2.49,
    'In': 2.43, 'Sn': 2.42, 'Sb': 2.47, 'Te': 1.99, 'I': 2.04, 'Xe': 2.06,
    'Cs': 3.48, 'Ba': 3.03, 'La': 2.98, 'Ce': 2.88, 'Pr': 2.92, 'Nd': 2.95,
    'Sm': 2.90, 'Eu': 2.87, 'Gd': 2.83, 'Tb': 2.79, 'Dy': 2.87, 'Ho': 2.81,
    'Er': 2.83, 'Tm': 2.79, 'Yb': 2.80, 'Lu': 2.74, 'Hf': 2.63, 'Ta': 2.53,
    'W': 2.57, 'Re': 2.49, 'Os': 2.48, 'Ir': 2.41, 'Pt': 2.29, 'Au': 2.32,
    'Hg': 2.45, 'Tl': 2.47, 'Pb': 2.60, 'Bi': 2.54, 'Ac': 2.8, 'Th': 2.93,
    'Pa': 2.88, 'U': 2.71, 'Np': 2.82, 'Pu': 2.81, 'Am': 2.83, 'Cm': 3.05,
    'Bk': 3.4, 'Cf': 3.05, 'Es': 2.7
}

_CSD_covalent_radius = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
    'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
    'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
    'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92,
    'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
    'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36,
    'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
    'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06,
    'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69
}

_periodic_table = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
    'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47,
    'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69,
    'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
    'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102,
    'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
    'Uut': 113, 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118
}

_nonmetal_element = {
    'H': 1, 'He': 2, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
    'As': 33, 'Se': 34, 'Br': 35, 'Te': 52, 'I': 53,
    'At': 85, 'Ne': 10, 'Ar': 18, 'Kr': 36, 'Xe': 54,
    'Rn': 86, 'Uus': 117, 'Uuo': 118
}

_metal_element = {
    'Li': 3, 'Be': 4, 'Na': 11, 'Mg': 12, 'Al': 13, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
    'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76,
    'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'Fr': 87, 'Ra': 88, 'Ac': 89,
    'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Uut': 113, 'Fl': 114, 'Uup': 115, 'Lv': 116
}


def atomic_radius(element):
    """
    Get atomic radius.

    Parameter
    ------
    element : 'str'
        Element symbol.

    Returns
    ------
    'float'
        Atomic radius in float.

    """
    return _vdw_radius[element]


class Atom:
    """
    class 'atom'.

    """

    def __init__(
            self,
            atom_id,
            element,
            atomic_number,
            degree,
            is_aromatic,
            hybridization,
            valence=0,
            formal_charge=0):
        """
        Initialize a class 'Atom' instance.

        Parameter
        ---------
        atom_id : 'int'
            ID of this atom.
        element : 'str'
            Element symbol.
        atomic_number : 'int'
            Atomic number.
        degree : 'int'
            Degree of this atom.
        is_aromatic : 'bool'
            Whether this atom is aromatic.
        hybridization : 'str'
            Hybridization of this atom.
        valence : 'int' , default = 0
            Valence of this atom.
        formal_charge : 'int', default = 0
            Formal charge of this atom.

        """
        self._element = element
        self._radius = atomic_radius(element)
        self._id = atom_id
        self._valence = valence
        if element in _nonmetal_element.keys():
            self._is_metal = False
        else:
            self._is_metal = True
        self._hybridization = hybridization
        self._formal_charge = formal_charge
        self._atomic_number = atomic_number
        self._is_aromatic = is_aromatic
        self._degree = degree

    def get_atomic_radius(self):
        """
        Get atomic radius.

        Returns
        ------
        atomic_radius: 'float'

        """
        return self._radius

    def get_atomic_number(self):
        """
        Get atomic number.

        Returns
        ------
        atomic_number: 'int'

        """
        return self._atomic_number

    def get_element(self):
        """
        Get element string.

        Returns
        ------
        element_str: 'str'

        """
        return self._element

    def get_atom_id(self):
        """
        Get the id of a specified atom.

        Returns
        ------
        'int'

        """
        return self._id

    # def get_atom_position(self):
    #     """
    #     Get the position of a specified atom.
    #
    #     Returns
    #     ------
    #     'numpy.ndarray'
    #
    #     """
    #     return self._atom_position

    def get_valence(self):
        """
        Get the valence of a specified atom.

        Returns
        ------
        'int'

        """
        return self._valence

    def get_covalent_radius(self):
        """
        Get the covalent radius of a specified atom.

        """
        return _CSD_covalent_radius[self._element]

    def get_formal_charge(self):
        """
        Get the formal charge of a specified atom.

        """
        return self._formal_charge

    def reset_atom_id(self, new_atom_id):
        """
        Reset the id of a specified atom.

        """
        self._id = new_atom_id

    def reset_formal_charge(self, new_formal_charge):
        """
        Set the formal charge of a specified atom.

        """
        self._formal_charge = new_formal_charge

    def reset_valence(self, new_valence):
        """
        Set the valence of a specified atom.

        """
        self._valence = new_valence

    def get_degree(self):
        return self._degree

    def get_is_aromatic(self):
        return self._is_aromatic

    def get_hybridization(self):
        return self._hybridization

    def get_is_metal(self):
        return self._is_metal
