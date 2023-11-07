import os
import numpy as np
import networkx as nx

"""
Write Gaussian output file
"""

element_weights = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'K': 19,
    'Ca': 20,
    'Sc': 21,
    'Ti': 22,
    'V': 23,
    'Cr': 24,
    'Mn': 25,
    'Fe': 26,
    'Ni': 27,
    'Cu': 28,
    'Zn': 29,
    'Ga': 30,
    'Ge': 31,
    'As': 32,
    'Se': 33,
    'Br': 34,
    'Kr': 35,
    'Rb': 36,
    'Sr': 37,
    'Y': 38,
    'Zr': 39,
    'Nb': 40,
    'Mo': 41,
    'Tc': 42,
    'Ru': 43,
    'Rh': 44,
    'Pd': 45,
    'Ag': 46,
    'Cd': 47,
    'In': 48,
    'Sn': 49,
    'Sb': 50,
    'Te': 51,
    'I': 52
}


class GaussianOutput:
    def __init__(self, mol, nproc=48, mem='60GB', chk=None, method='B3LYP', basis='Def2SVP',
                 Dispersion='GD3BJ', opt=True, readopt=True, not_atoms='C', extra_section=None, title='Gaussian Calulation', charge=0,
                 multiplicity=1, basis_from_bse=False, bse=None, nuopt_frag=1):
        self._mol = mol
        self._nproc = nproc
        self._mem = mem
        self._chk = chk
        self._method = method
        self._basis = basis
        self._Dispersion = Dispersion
        self._readopt = readopt
        self._opt = opt
        self._not_atoms = not_atoms
        self._title = title
        self._charge = charge
        self._multiplicity = multiplicity
        self._extra_section = extra_section
        self._num_frag = len(self._define_frag_index())
        self._atom_set = set(i.get_element() for i in mol.get_atoms())
        self._sorted_symbols = sorted(self._atom_set, key=lambda symbol: element_weights.get(symbol))
        self._basis_from_bse = basis_from_bse
        self._bse = bse
        if self._num_frag > 1:
            pass

    def _define_frag_index(self):
        """
        Return a list of list, each list contains the index of atoms in a fragment.
        """
        graph = nx.Graph()

        # Let atoms be the nodes of the graph.
        for atom in self._mol.get_atoms():
            graph.add_node(atom.get_atom_id())

        # Bonds are the edges of the graph.
        for bond in self._mol.get_bonds():
            e = (bond.get_atom_1_id(), bond.get_atom_2_id())
            graph.add_edge(*e)

        component_idx = []

        for component in nx.connected_components(graph):
            component_idx.append(list(component))
        return component_idx

    def _write_to_com(self):
        info = []
        # Write header
        info.append(f'%nprocshared = {self._nproc}\n')
        info.append(f'%mem = {self._mem}\n')
        if self._chk is not None:
            info.append(f'%chk={self._chk}\n')
        else:
            raise ValueError('chk file is not specified.')
        info.append(f'#p {self._method}/{self._basis} EmpiricalDispersion={self._Dispersion} ')
        if self._readopt and not self._opt:
            info[-1] += 'opt=readopt '
        if self._opt and not self._readopt:
            info[-1] += 'opt '
        if self._num_frag > 1:
            info[-1] += f'counterpoise={self._num_frag} '
        if self._extra_section is not None:
            info[-1] += self._extra_section
        info[-1] += '\n\n'
        info.append(f'{self._title}\n\n')
        info.append(f'{self._charge} {self._multiplicity}\n')

        positions = self._mol.get_positions()

        # Write atoms
        if self._num_frag == 1:
            for i, atom in enumerate(self._mol.get_atoms()):
                x = positions[i][0]
                y = positions[i][1]
                z = positions[i][2]
                # 将x、y、z坐标保留到小数点后8位（包含0）
                info.append(f'{atom.get_element():<2}                 {x:<.8f}   {y:<.8f}   {z:<.8f}\n')
            if self._readopt:
                info.append('\n')
                info.append(f'notatoms={self._not_atoms}\n')

            if self._basis_from_bse:
                info.append('\n')
                for a in self._sorted_symbols:
                    info.extend(self._bse[a])
                    info.append('****\n')
            info.append('\n\n\n\n\n\n')
            return info

        elif self._num_frag > 1:
            self._num_frag = len(self._define_frag_index())
            for i, atom in enumerate(self._mol.get_atoms()):
                x = positions[i][0]
                y = positions[i][1]
                z = positions[i][2]
                frag_idx = 0
                for j in range(self._num_frag):
                    if atom.get_atom_id() in self._define_frag_index()[j]:
                        frag_idx = j + 1
                # 将x、y、z坐标保留到小数点后8位（包含0）
                info.append(f'{atom.get_element():<2}                 {x:<.8f}   {y:<.8f}   {z:<.8f}    {frag_idx}\n')
            if self._readopt:
                info.append('\n')
                info.append(f'notatoms={self._not_atoms}\n')
            if self._basis_from_bse:
                for a in self._sorted_symbols:
                    info.append('\n')
                    info.extend(self._bse[a])
                    info.append('****\n')
            info.append('\n\n\n\n\n\n')
            return info
        else:
            raise ValueError('Number of fragments should be greater than 0.')

    def write_com(self, path, name):
        if name[-4:] != '.com':
            name += '.com'
        if os.path.exists(f'{path}/{name}'):
            os.remove(f'{path}/{name}')
        f = open(f'{path}/{name}', 'w')
        f.writelines(self._write_to_com())
        f.close()
        print(f'Gaussian input file {name} has been written to {path}.')
