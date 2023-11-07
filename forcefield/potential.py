# -*- coding: utf-8 -*-
# @Time    : 6/13/22 8:18 PM
# @Author  : name
# @File    : potential.py
"""
Future work:
    1_Energy_Contrast. Convert 'guest' to 'list(guest)' for multiple guest.
"""
import numpy as np

from numpy import ndarray
from scipy.spatial import distance
from rdkit.Chem import AllChem

from .obabel_parameter_assign import OBParameterAssign
from .amber_parameters import Amber99Parameter
from ..molecule import HostGuestComplex
from .uff4mof_parameter_assign import UFF4MOFAssign

from scipy.spatial import distance
from rdkit.Chem import AllChem


class SimpleLJPotential:

    def __init__(self, host, guest, epsilon=1):
        """

        """
        self._host = host
        self._guest = guest
        self._epsilon = epsilon
        self._host_positions = self._host.get_positions()
        self._guest_positions = self._guest.get_positions()

    @staticmethod
    def _lorentz_berthelot_method(sigma_i: float, sigma_j: float) -> float:
        return 0.5 * (sigma_i + sigma_j)

    def _cal_sigma(self):
        host_radii = self._host.get_radii_list()
        guest_radii = self._guest.get_radii_list()

        sigma_matrix = np.zeros((self._host.get_atom_number(), self._guest.get_atom_number()))
        for i in range(self._host.get_atom_number()):
            for j in range(self._guest.get_atom_number()):
                sigma_matrix[i, j] = self._lorentz_berthelot_method(
                    host_radii[i], guest_radii[j]
                )

        return sigma_matrix

    def _cal_dis_matrix(self):
        return distance.cdist(self._host_positions, self._guest_positions)

    def _cal_lj_potential(self, sigma, distance, epsilon):
        return 4 * epsilon * (
                (sigma / distance) ** 12 - (sigma / distance) ** 6
        )

    def cal_potential(self):
        pair_potential = []
        sigma_matrix = self._cal_sigma()
        dis_matrix = self._cal_dis_matrix()

        for i in range(self._host.get_atom_number()):
            for j in range(self._guest.get_atom_number()):
                pair_potential.append(
                    self._cal_lj_potential(
                        sigma=sigma_matrix[i, j],
                        distance=dis_matrix[i, j],
                        epsilon=self._epsilon,
                    )
                )

        return np.sum(pair_potential, axis=0)


class AmberPotential:
    """
    Inspired by the article:
    'EDock: blind protein–ligand docking by replica-exchange monte carlo simulation'

    Notes
    ------
    There are now 3 equations in this class.
    1_Energy_Contrast. Van Der Waals interaction
        sum(w1 * ((A_ij / d_ij ** 12) - (B_ij / d_ij ** 6)))
        ------
        where:
            A_ij = epsilon_ij * (R_ij ** 12) ;
            B_ij = 2 * epsilon_ij * (R_ij ** 6) ;
            R_ij = host_radii[i] + guest_radii[j] ;
            epsilon_ij = (epsilon_i * epsilon_j) ** 0.5;
            d_ij = dis_matrix[i, j]

    2. Electrostatic interaction
        sum(w2 * ((host_charge[i] * guest_charge[j]) / 4 * dis_matrix[i, j]))

    3. Constrain the distance between host and guest

    """

    def __init__(
            self,
            host,
            guest,
            host_par, guest_par
    ):
        """
        host_par: The parameter of host.
            In the form of [host_atom_num, radii, epsilon, charge], where host_atom_num is "int",
            radii, epsilon, charge are "list".
        guest_par: The parameter of guest.
            In the form of [guest_atom_num, radii, epsilon, charge], where guest_atom_num is "int",
            radii, epsilon, charge are "list".
        """
        self._host = host
        self._guest = guest
        self._host_par = host_par
        self._guest_par = guest_par

    def _get_charge_param(self):
        pass

    def _get_pair_distance(self):
        dis_matrix = distance.cdist(
            self._host.get_positions(),
            self._guest.get_positions()
        )
        return dis_matrix

    def _cal_vdw(self,
                 host_radii: list,
                 guest_radii: list,
                 len_host: int,
                 len_guest: int,
                 epsilon_i: list,
                 epsilon_j: list,
                 dis_matrix: ndarray) -> ndarray:
        vdw_array = np.zeros((len_host, len_guest))

        for i in range(len_host):
            for j in range(len_guest):
                epsilon_ij = (epsilon_i[i] * epsilon_j[j]) ** 0.5
                R_ij = host_radii[i] + guest_radii[j]
                A_ij = epsilon_ij * (R_ij ** 12)
                B_ij = 2 * epsilon_ij * (R_ij ** 6)
                d_ij = dis_matrix[i, j]

                vdw_array[i, j] = (
                        ((A_ij / d_ij ** 12) - (B_ij / d_ij ** 6))
                )

        return vdw_array

    def _cal_electrostatic(self,
                           len_host: int,
                           len_guest: int,
                           host_charge: list,
                           guest_charge: list,
                           dis_matrix: ndarray) -> ndarray:
        charge_interaction = np.zeros((len_host, len_guest))

        for i in range(len_host):
            for j in range(len_guest):
                charge_interaction[i, j] = (
                        ((host_charge[i] * guest_charge[j]) / 4 * dis_matrix[i, j]))
        return charge_interaction

    def cal_potential(self):
        dis_matrix = self._get_pair_distance()
        len_host = self._host_par[0]
        len_guest = self._guest_par[0]

        radii_i, epsilon_i, charge_i = self._host_par[1], self._host_par[2], self._host_par[3]
        radii_j, epsilon_j, charge_j = self._guest_par[1], self._guest_par[2], self._guest_par[3]

        vdw_potential = self._cal_vdw(
            host_radii=radii_i,
            guest_radii=radii_j,
            len_host=len_host,
            len_guest=len_guest,
            epsilon_i=epsilon_i,
            epsilon_j=epsilon_j,
            dis_matrix=dis_matrix
        )

        electrostatic_potential = self._cal_electrostatic(
            len_host=len_host,
            len_guest=len_guest,
            host_charge=charge_i,
            guest_charge=charge_j,
            dis_matrix=dis_matrix
        )

        return np.sum(vdw_potential) + np.sum(electrostatic_potential)


class UFFPotential:
    def __init__(self, host, guest, host_par, guest_par, scratch_dir='./scratch'):
        """
        host_par: The parameter of host.
            In the form of [host_atom_num, radii, epsilon, charge], where host_atom_num is "int",
            radii, epsilon, charge are "list".
        guest_par: The parameter of guest.
            In the form of [guest_atom_num, radii, epsilon, charge], where guest_atom_num is "int",
            radii, epsilon, charge are "list".
        """
        self._host = host
        self._guest = guest
        self._complex = HostGuestComplex.init_from_molecule(self._host, self._guest)
        self._ff = 'uff'
        self._scratch = scratch_dir
        self._host_par = host_par
        self._guest_par = guest_par

    @staticmethod
    def _cal_vdw(
            len_host,
            len_guest,
            D_i,
            D_j,
            x_i,
            x_j,
            dis_matrix
    ):
        """
        Calculate van der Waals energy.

        Parameters
        -----------------


        Returns
        -----------------
        'numpy.ndarray': The van der Waals energy of each pair of atoms.
                         Unit: kcal/mol.

        """
        vdw_potential = np.zeros(shape=(len_host, len_guest))
        for i in range(len_host):
            for j in range(len_guest):
                D_ij = np.sqrt(D_i[i] * D_j[j])
                x_ij = np.sqrt(x_i[i] * x_j[j])
                d_ij = dis_matrix[i, j]
                vdw_potential[i, j] = D_ij * (-2 * (x_ij / d_ij) ** 6 + (x_ij / d_ij) ** 12)

        return vdw_potential

    @staticmethod
    def _cal_electrostatic(
            len_host,
            len_guest,
            q_i,
            q_j,
            dis_matrix
    ):
        """
        Calculate electrostatic energy.

        Parameters
        -----------------
        """
        electrostatic_potential = np.zeros(shape=(len_host, len_guest))
        for i in range(len_host):
            for j in range(len_guest):
                q_ij = q_i[i] * q_j[j]
                d_ij = dis_matrix[i, j]
                electrostatic_potential[i, j] = 332.0637 * (q_ij / d_ij)

        return electrostatic_potential

    def cal_potential(self):
        len_host = self._host_par[0]
        len_guest = self._guest_par[0]

        D_i, x_i, q_i = self._host_par[1], self._host_par[2], self._host_par[3]
        D_j, x_j, q_j = self._guest_par[1], self._guest_par[2], self._guest_par[3]

        dis_matrix = distance.cdist(
            self._host.get_positions(),
            self._guest.get_positions()
        )

        vdw_potential = self._cal_vdw(
            len_host=len_host,
            len_guest=len_guest,
            D_i=D_i,
            D_j=D_j,
            x_i=x_i,
            x_j=x_j,
            dis_matrix=dis_matrix
        )

        electrostatic_potential = self._cal_electrostatic(
            len_host=len_host,
            len_guest=len_guest,
            q_i=q_i,
            q_j=q_j,
            dis_matrix=dis_matrix
        )

        return np.sum(vdw_potential) + np.sum(electrostatic_potential)


class UFF4MOFPotential:
    def __init__(self, host, guest, host_par, guest_par):
        self._host = host
        self._guest = guest
        self._complex = HostGuestComplex.init_from_molecule(self._host, self._guest)
        self._host_par = host_par
        self._guest_par = guest_par

    @staticmethod
    def _cal_vdw(
            len_host,
            len_guest,
            D_i,
            D_j,
            x_i,
            x_j,
            dis_matrix
    ):
        """
        Calculate van der Waals energy.

        Parameters
        -----------------


        Returns
        -----------------
        'numpy.ndarray': The van der Waals energy of each pair of atoms.
                         Unit: kcal/mol.

        """
        vdw_potential = np.zeros(shape=(len_host, len_guest))
        for i in range(len_host):
            for j in range(len_guest):
                D_ij = np.sqrt(D_i[i] * D_j[j])
                x_ij = np.sqrt(x_i[i] * x_j[j])
                d_ij = dis_matrix[i, j]
                vdw_potential[i, j] = D_ij * (-2 * (x_ij / d_ij) ** 6 + (x_ij / d_ij) ** 12)

        return vdw_potential

    @staticmethod
    def _cal_electrostatic(
            len_host,
            len_guest,
            q_i,
            q_j,
            dis_matrix
    ):
        """
        Calculate electrostatic energy.

        Parameters
        -----------------
        """
        electrostatic_potential = np.zeros(shape=(len_host, len_guest))
        for i in range(len_host):
            for j in range(len_guest):
                q_ij = q_i[i] * q_j[j]
                d_ij = dis_matrix[i, j]
                electrostatic_potential[i, j] = 332.0637 * (q_ij / d_ij)

        return electrostatic_potential

    def cal_potential(self):
        len_host = self._host_par[0]
        len_guest = self._guest_par[0]

        D_i, x_i, q_i = self._host_par[1], self._host_par[2], self._host_par[3]
        D_j, x_j, q_j = self._guest_par[1], self._guest_par[2], self._guest_par[3]

        dis_matrix = distance.cdist(
            self._host.get_positions(),
            self._guest.get_positions()
        )

        vdw_potential = self._cal_vdw(
            len_host=len_host,
            len_guest=len_guest,
            D_i=D_i,
            D_j=D_j,
            x_i=x_i,
            x_j=x_j,
            dis_matrix=dis_matrix
        )

        electrostatic_potential = self._cal_electrostatic(
            len_host=len_host,
            len_guest=len_guest,
            q_i=q_i,
            q_j=q_j,
            dis_matrix=dis_matrix
        )

        return np.sum(vdw_potential) + np.sum(electrostatic_potential)
