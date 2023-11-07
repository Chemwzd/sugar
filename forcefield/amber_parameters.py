# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 下午2:49
# @Author  : wzd
# @File    : amber_parameters.py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Any

from numpy import ndarray
from ..molecule import Molecule, HostMolecule, GuestMolecule, HostGuestComplex


class ElementNotFoundError(Exception):
    pass


class Amber99Parameter:
    # Parameters for the van der Waals energy potential.
    # Taken from AMBER99.
    # Reference:  J.Comput.Chem., 21:1049–1074.

    Amber99_element = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'Si', 'I', 'K', 'Na', 'Mg', 'Li', 'Rb', 'Cs',
                       'Ca', 'Zn', 'Fe', 'Sr', 'Ba', 'V', 'Du']
    single_para_elements = ['N', 'S', 'P', 'F', 'Cl', 'Br', 'Si', 'I', 'K', 'Na', 'Mg', 'Li', 'Rb', 'Cs',
                            'Ca', 'Zn', 'Fe', 'Sr', 'Ba', 'V', 'Du']

    def __init__(self):
        pass

    def _get_parameters(
            self,
            molecule: Optional[Molecule],
            draw=False
    ) -> nx.Graph:
        Amber99_LJ_Parameters = {
            'C': [1.908, 0.086],
            'C.3': [1.908, 0.1094],
            'H': [0.6, 0.0157],
            'H(O)': [0.1, 0.001],
            'H(S)': [0.6, 0.0157],
            'H(C.3)': [1.487, 0.0157],
            'H(C.3(2O/2N/2S/(N)(O)/(S)(O)/(S)(N)(2F)/(2Cl)/(2Br)))': [1.287, 0.0157],
            'H(C.3(O/N/S/F/Cl/Br))': [1.387, 0.0157],
            'H(C.3(N.4))': [1.1, 0.0157],
            'H(C.ar/C.2)': [1.459, 0.015],
            'H(C.ar(O/N)) or H(C.2(O/N))': [1.409, 0.015],
            'H(C.ar(2O/2N/(N)(O))) or H(C.2(2O)/(2N)/(N)(O))': [1.359, 0.015],
            'H(C.1_Energy_Contrast)': [1.459, 0.015],
            'N': [1.827, 0.17],
            'O': [1.6612, 0.21],
            'O.3': [1.6837, 0.17],
            'O.3(H)': [1.721, 0.2104],
            'O.3(2H)': [1.7683, 0.152],
            'S': [2, 0.25],
            'P': [2.1, 0.2],
            'F': [1.75, 0.061],
            'Cl': [1.948, 0.265],
            'Br': [2.22, 0.32],
            'Si': [2.22, 0.32],
            'I': [2.35, 0.4],
            'K': [2.658, 0.000328],
            'Na': [1.868, 0.00277],
            'Mg': [0.787, 0.875],
            'Li': [1.137, 0.0183],
            'Rb': [2.956, 0.00017],
            'Cs': [3.395, 0.0000806],
            'Ca': [1.326, 0.4497],
            'Zn': [1.1, 0.0125],
            'Fe': [1.2, 0.05],
            'Sr': [1.742, 0.118],
            'Ba': [2.124, 0.047],
            'V': [2.1, 0.32],
            'Du': [0, 0]
        }
        atoms = [atom for atom in molecule.get_atoms()]

        # Define molecule into graph.
        graph = nx.Graph()
        for atom in atoms:
            graph.add_node(atom.get_atom_id(), element=atom.get_element())
        for bond in molecule.get_bonds():
            edge = (bond.get_atom_1_id(), bond.get_atom_2_id())
            graph.add_edge(*edge, bond_id=bond.get_bond_id(), bond_type=bond.get_bond_type())

        # Determine atom parameters.
        # ------FUTURE------
        # store atom parameters in a list directly maybe faster.
        # ------------------
        for i in range(graph.number_of_nodes()):
            if graph.nodes[i]['element'] in self.single_para_elements:
                graph.nodes[i]['radii'] = Amber99_LJ_Parameters[graph.nodes[i]['element']][0]
                graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters[graph.nodes[i]['element']][1]
            else:
                if graph.nodes[i]['element'] == 'C':
                    if len([n for n in graph.neighbors(i)]) == 4:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['C.3'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['C.3'][1]
                    else:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['C'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['C'][1]

                elif graph.nodes[i]['element'] == 'H':
                    neighbors = [n for n in graph.neighbors(i)]
                    neighbor_element = graph.nodes[neighbors[0]]['element']
                    if 'O' == neighbor_element:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['H(O)'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['H(O)'][1]
                    elif 'S' == neighbor_element:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['H(S)'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['H(S)'][1]
                    elif 'C' == neighbor_element:
                        # id of C atom.
                        H_neighbor_id = neighbors[0]
                        # Determine hybrid state of C atom neighbored with H atom.
                        # C atom can be classified into 3 groups : sp3 / sp2 and sp.
                        C_neighbors = [n for n in graph.neighbors(H_neighbor_id)]
                        # How many atoms connect to C atom.
                        C_neighbor_number = len(C_neighbors)
                        # Atoms' symbol around C.
                        C_neighbor_elements = [graph.nodes[n]['element'] for n in C_neighbors]
                        # ------TEST------
                        # print(f"{H_neighbor_id} neighbor: {C_neighbor_elements}")
                        # sp.
                        if C_neighbor_number == 2:
                            graph.nodes[i]['radii'] = Amber99_LJ_Parameters['H(C.1_Energy_Contrast)'][0]
                            graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['H(C.1_Energy_Contrast)'][1]
                        # sp2.
                        elif C_neighbor_number == 3:
                            # 3 conditions considered in sp2 C.
                            if 'O' or 'N' in C_neighbor_elements:
                                # Check the number of O or N.
                                # O or N hit twice.
                                if C_neighbor_elements.count('O') or C_neighbor_elements.count(
                                        'N') == 2 or 'O' and 'N' in C_neighbor_elements:
                                    graph.nodes[i]['radii'] = \
                                        Amber99_LJ_Parameters['H(C.ar(2O/2N/(N)(O))) or H(C.2(2O)/(2N)/(N)(O))'][0]
                                    graph.nodes[i]['epsilon'] = \
                                        Amber99_LJ_Parameters['H(C.ar(2O/2N/(N)(O))) or H(C.2(2O)/(2N)/(N)(O))'][1]
                                else:
                                    graph.nodes[i]['radii'] = \
                                        Amber99_LJ_Parameters['H(C.ar(O/N)) or H(C.2(O/N))'][0]
                                    graph.nodes[i]['epsilon'] = \
                                        Amber99_LJ_Parameters['H(C.ar(O/N)) or H(C.2(O/N))'][1]
                            else:
                                graph.nodes[i]['radii'] = Amber99_LJ_Parameters['H(C.ar/C.2)'][0]
                                graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['H(C.ar/C.2)'][1]
                        # sp3 C.
                        # new_version
                        else:
                            elements = ['O', 'N', 'S', 'Cl', 'Br']
                            for symbol in elements:
                                if symbol in C_neighbor_elements:
                                    # Check the number of elements above.
                                    # Hit twice.
                                    if not (not C_neighbor_elements.count('O') and not C_neighbor_elements.count(
                                            'N')) or C_neighbor_elements.count('S') or C_neighbor_elements.count(
                                        'F') or C_neighbor_elements.count('Cl') or C_neighbor_elements.count(
                                        'Br') == 2 or 'O' and 'N' or 'S' and 'O' or 'S' and 'N' in C_neighbor_elements:
                                        graph.nodes[i]['radii'] = \
                                            Amber99_LJ_Parameters[
                                                'H(C.3(2O/2N/2S/(N)(O)/(S)(O)/(S)(N)(2F)/(2Cl)/(2Br)))'][0]
                                        graph.nodes[i]['epsilon'] = \
                                            Amber99_LJ_Parameters[
                                                'H(C.3(2O/2N/2S/(N)(O)/(S)(O)/(S)(N)(2F)/(2Cl)/(2Br)))'][1]
                                    else:
                                        graph.nodes[i]['radii'] = \
                                            Amber99_LJ_Parameters['H(C.3(O/N/S/F/Cl/Br))'][0]
                                        graph.nodes[i]['epsilon'] = \
                                            Amber99_LJ_Parameters['H(C.3(O/N/S/F/Cl/Br))'][1]
                                else:
                                    graph.nodes[i]['radii'] = Amber99_LJ_Parameters['H(C.3)'][0]
                                    graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['H(C.3)'][1]
                    else:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['H'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['H'][1]
                elif graph.nodes[i]['element'] == 'O':
                    neighbors = [n for n in graph.neighbors(i)]
                    neighbor_element = [graph.nodes[i]['element'] for i in neighbors]
                    if neighbor_element.count('H') == 2:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['O.3(2H)'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['O.3(2H)'][1]
                    elif neighbor_element.count('H') == 1:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['O.3(H)'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['O.3(H)'][1]
                    elif neighbor_element.count('C') == 2:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['O.3'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['O.3'][1]
                    else:
                        graph.nodes[i]['radii'] = Amber99_LJ_Parameters['O'][0]
                        graph.nodes[i]['epsilon'] = Amber99_LJ_Parameters['O'][1]
        if draw:
            nx.draw(graph, with_labels=True)
            plt.show()
        return graph

    def get_parameters(
            self,
            molecule: Molecule
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Return the parameters of the molecule.

        Parameters
        ----------
        molecule : 'Molecule'
            The molecule to be analyzed.

        Returns
        ----------
        'list'
            The parameters of the molecule.

        Raises
        ----------
        'ElementNotFoundError'
            If the element is not found in the parameter list.

        """
        atoms = [atom for atom in molecule.get_atoms()]

        for atom in atoms:
            if atom.get_element() not in self.Amber99_element:
                raise ElementNotFoundError('Element {} is not found in Amber99.'.format(atom))

        # Calculate the parameters of the molecule.
        graph = self._get_parameters(molecule)
        radii = [graph.nodes[i]['radii'] for i in graph.nodes]
        epsilon = [graph.nodes[i]['epsilon'] for i in graph.nodes]
        charge = molecule.cal_gasteiger_charge()

        return radii, epsilon, charge




