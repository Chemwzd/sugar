import os
import numpy as np
import networkx as nx
import pandas as pd

from numpy import ndarray
from scipy.spatial import distance
from rdkit.Chem import AllChem
from openbabel import openbabel
from openbabel.openbabel import OBChargeModel
from .uff_parameter import UFF


class OBParameterAssign:

    def __init__(
            self,
            molecule,
            scratch_dir='./scratch',
            ff_name='uff',
            charge='qeq'
    ):
        if not os.path.exists(scratch_dir):
            os.mkdir(scratch_dir)

        molecule.write_to_mol_file(
            scratch_dir,
            'obpotential_scratch.mol',
            True
        )

        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat('.mol')
        ob_mol = openbabel.OBMol()
        ob_conversion.ReadFile(ob_mol, f"{scratch_dir}/obpotential_scratch.mol")

        ff = openbabel.OBForceField.FindForceField(ff_name)
        ff.Setup(ob_mol)
        ff.GetAtomTypes(ob_mol)
        ff.GetCoordinates(ob_mol)

        charge_model = OBChargeModel.FindType(charge)
        charge_model.ComputeCharges(ob_mol)
        partial_charges = charge_model.GetPartialCharges()

        self._atom_type = [i.GetData('FFAtomType').GetValue() for i in openbabel.OBMolAtomIter(ob_mol)]
        self._partial_charges = partial_charges
        self._mol = molecule
        os.system(f"rm -r {scratch_dir}")

    def get_atom_type(self):
        return self._atom_type

    def get_charge(self):
        return self._partial_charges

    def mol_to_graph(self):
        """
        Convert the molecule to a graph,
        and assign atom type for UFF force field.

        Returns
        ------------
        'networkx.Graph': The graph of the molecule.

        """
        mol_graph = nx.Graph()
        atoms = [i for i in self._mol.get_atoms()]

        atom_type = self.get_atom_type()
        charge = self.get_charge()

        for i in range(self._mol.get_atom_number()):
            mol_graph.add_node(i, element=atoms[i].get_element(), uff_atom_type=atom_type[i],
                               partial_charge=charge[i], D1=UFF[atom_type[i]][3],
                               x1=UFF[atom_type[i]][2])
        for bond in self._mol.get_bonds():
            edge = (bond.get_atom_1_id(), bond.get_atom_2_id())
            mol_graph.add_edge(*edge, bond_id=bond.get_bond_id(), bond_type=bond.get_bond_type())
        return mol_graph
