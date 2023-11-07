import networkx as nx
import os

from openbabel import openbabel
from openbabel.openbabel import OBChargeModel
from rdkit.Chem.rdchem import HybridizationType

from .uff4mof_parameter import UFF4MOF_ELEMENTS, get_uff4mof_para


class AtomTypeError(Exception):
    pass


class UFF4MOFAssign:

    def __init__(
            self,
            mol_path,
            format_in,
            molecule,
            ff_name='UFF',
            charge='qeq'
    ):
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat(str(format_in))
        ob_mol = openbabel.OBMol()

        if format_in == "smi":
            ob_conversion.ReadString(ob_mol, mol_path)
        elif format_in == "mol" or "xyz":
            ob_conversion.ReadFile(ob_mol, mol_path)
        else:
            raise ValueError(f"Unsupported file format : {format_in}.")

        ff = openbabel.OBForceField.FindForceField(ff_name)
        ff.Setup(ob_mol)
        ff.GetAtomTypes(ob_mol)
        ff.GetCoordinates(ob_mol)

        charge_model = OBChargeModel.FindType(charge)
        charge_model.ComputeCharges(ob_mol)
        partial_charges = charge_model.GetPartialCharges()

        self._uff_atom_type = [i.GetData('FFAtomType').GetValue() for i in openbabel.OBMolAtomIter(ob_mol)]
        self._partial_charges = partial_charges
        self._mol = molecule

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

        for i in range(self._mol.get_atom_number()):
            mol_graph.add_node(i, element=atoms[i].get_element(),
                               formal_charge=atoms[i].get_formal_charge(), valence=atoms[i].get_valence(),
                               hybridization=atoms[i].get_hybridization(),
                               is_metal=atoms[i].get_is_metal(), uff_atom_type=self._uff_atom_type[i],
                               partial_charge=self._partial_charges[i])
        for bond in self._mol.get_bonds():
            edge = (bond.get_atom_1_id(), bond.get_atom_2_id())
            mol_graph.add_edge(*edge, bond_id=bond.get_bond_id(), bond_type=bond.get_bond_type())
        return mol_graph

    def assign_uff4mof(self):
        mol_graph = self.mol_to_graph()
        for node in mol_graph.nodes:
            element = mol_graph.nodes[node]['element']
            atom_type = mol_graph.nodes[node]['uff_atom_type']
            valence = mol_graph.nodes[node]['valence']
            hybridization = mol_graph.nodes[node]['hybridization']
            formal_charge = mol_graph.nodes[node]['formal_charge']

            if element in UFF4MOF_ELEMENTS:
                if element == 'O':
                    metal_flag = [mol_graph.nodes[n]['is_metal'] for n in mol_graph.neighbors(node)]
                    neighbor_element = [mol_graph.nodes[n]['element'] for n in mol_graph.neighbors(node)]
                    if atom_type == 'O_3':
                        if True in metal_flag:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'O_3_f'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type
                    elif atom_type == 'O_2':
                        if 'Si' in neighbor_element:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'O_2_z'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type
                    else:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type

                elif element == 'Al':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Al6+3'
                    elif atom_type == 'Al3':
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Al3f2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Sc':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Sc6+3'
                    elif atom_type == 'Sc3+3':
                        mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Ti':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ti4+2'
                    elif atom_type == 'Ti3+4':
                        mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'V':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'V_4+2'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'V_6+3'
                    elif hybridization == HybridizationType.SP3 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'V_3f2'
                    elif atom_type == 'V_3+5':
                        mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Cr':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cr4+2'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cr6f3'
                    elif atom_type == 'Cr6+3':
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cr6f3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Mn':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mn6+3'
                    elif hybridization == HybridizationType.SP3 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mn3f2'
                    elif valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mn4+2'
                    elif hybridization == HybridizationType.SP and formal_charge == 1:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mn1f1'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mn8f4'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Fe':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Fe6+3'
                    elif valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Fe4+2'
                    else:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = mol_graph.nodes[node]['uff_atom_type']

                elif element == 'Co':
                    if valence == 4 and formal_charge == 2:
                        if hybridization == HybridizationType.SP3:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Co3+2'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Co4+2'
                    elif valence == 2 and formal_charge == 1:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Co1f1'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Co6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Cu':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cu4+2'
                    elif valence == 2 and formal_charge == 1:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cu1f1'
                    elif valence == 3 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cu2f2'
                    elif valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cu3f2'
                    elif valence == 4 and formal_charge == 1:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cu3+1_Energy_Contrast'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Zn':
                    if valence == 4 and formal_charge == 2:
                        if hybridization == HybridizationType.SP3:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Zn3f2'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Zn4+2'
                    elif valence == 2 and formal_charge == 1:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Zn1f1'
                    elif valence == 3 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Zn2f2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Li':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Li3f2'
                    else:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type

                elif element == 'Na':
                    if valence == 4:
                        if hybridization == HybridizationType.SP3:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Na3f2'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Na4f2'
                    else:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type

                elif element == 'Mg':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mg6f3'
                    elif valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mg3+2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'K':
                    if valence == 4:
                        if hybridization == HybridizationType.SP3:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'K_3f2'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'K_4f2'
                    else:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type

                elif element == 'Ca':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ca3f2'
                    elif valence == 6 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ca6+2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Ga':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ga3f2'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ga6f3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Sr':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Sr8f4'
                    elif valence == 6 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Sr6+2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Y':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Y_6f3'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Y_8f4'
                    elif valence == 4 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Y_3+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Zr':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Zr8f4'
                    elif valence == 4 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Zr3+4'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Nb':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Nb8f4'
                    elif valence == 4 and formal_charge == 5:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Nb3+5'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Mo':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mo8f4'
                    elif valence == 4:
                        if hybridization == HybridizationType.SP3:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mo3f2'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mo4f2'
                    elif valence == 4 and formal_charge == 6:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Mo3+6'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Tc':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Tc4f2'
                    elif valence == 6 and formal_charge == 5:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Tc6+5'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Ru':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ru4f2'
                    elif valence == 6 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ru6+2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Pd':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Pd6f3'
                    elif valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Pd4+2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Ag':
                    if valence == 2 and formal_charge == 1:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ag1f1'
                    elif valence == 3 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ag2f2'
                    elif valence == 4 and formal_charge == 2:
                        if hybridization == HybridizationType.SP3:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ag3f2'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ag4f2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Cd':
                    if valence == 2 and formal_charge == 1:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cd1f1'
                    elif valence == 4 and formal_charge == 2:
                        if hybridization == HybridizationType.SP3:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cd3f2'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cd4f2'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Cd8f4'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'In':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'In3f2'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'In6f3'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'In8f4'
                    elif valence == 4 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'In3+3'

                elif element == 'Ba':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ba3f2'
                    elif valence == 6 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ba6+2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'La':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'La8f4'
                    elif valence == 4 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'La3+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}.")

                elif element == 'Ce':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ce8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ce6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Pr':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Pr8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Pr6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Nd':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Nd8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Nd6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Sm':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Sm8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Sm6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Eu':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Eu6f3'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Eu8f4'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Gd':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Gd6f3'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Gd8f3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Tb':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Tb8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Tb6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Dy':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Dy8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Dy6f3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Ho':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ho8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Ho6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Er':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Er8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Er6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Tm':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Tm8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Tm6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Yb':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Yb6f3'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Yb8f4'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Lu':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Lu8f4'
                    elif valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Lu6+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Hf':
                    if valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Hf8f4'
                    elif valence == 4 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Hf3+4'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'W':
                    if valence == 4 and formal_charge == 2:
                        if hybridization == HybridizationType.SP3:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'W_3f2'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'W_4f2'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'W_8f4'
                    elif valence == 4 and formal_charge == 6:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'W_3+6'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Re':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Re6f3'
                    elif valence == 6 and formal_charge == 5:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Re6+5'
                    elif valence == 4 and formal_charge == 7:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Re3+7'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Os':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Os4f2'
                    elif valence == 6 and formal_charge == 6:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Os6+6'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Pt':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Pt4+2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Au':
                    if valence == 2 and formal_charge == 1:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Au1f1'
                    elif valence == 4 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Au4+3'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Hg':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Hg3f2'
                    elif valence == 2 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Hg1+2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'Pb':
                    if valence == 4 and formal_charge == 2:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'Pb4f2'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'U':
                    if valence == 6 and formal_charge == 3:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'U_6f3'
                    elif valence == 8 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'U_8f4'
                    elif valence == 6 and formal_charge == 4:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = 'U_6+4'
                    else:
                        raise AtomTypeError(
                            f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                        )

                elif element == 'S':
                    metal_flag = [mol_graph.nodes[n]['is_metal'] for n in mol_graph.neighbors(node)]
                    if atom_type == 'S_3+6':
                        if True in metal_flag:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = 'S_3_f'
                        else:
                            mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type
                    else:
                        mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type

                else:
                    raise AtomTypeError(
                        f"Atom {element} is not supported with valence {valence} and charge {formal_charge}."
                    )
            else:
                mol_graph.nodes[node]['uff4mof_atom_type'] = atom_type
        return mol_graph

    def get_parameter(self):
        """


        """
        mol_graph = self.assign_uff4mof()
        for node in mol_graph.nodes:
            para_list = get_uff4mof_para(mol_graph.nodes[node]['uff4mof_atom_type'])
            mol_graph.nodes[node]['r1'] = para_list[0]
            mol_graph.nodes[node]['theta0'] = para_list[1]
            mol_graph.nodes[node]['x1'] = para_list[2]
            mol_graph.nodes[node]['D1'] = para_list[3]
            mol_graph.nodes[node]['zeta'] = para_list[4]
            mol_graph.nodes[node]['Z1'] = para_list[5]
            mol_graph.nodes[node]['Vi'] = para_list[6]
            mol_graph.nodes[node]['Uj'] = para_list[7]
            mol_graph.nodes[node]['Xi'] = para_list[8]
            mol_graph.nodes[node]['Hard'] = para_list[9]
            mol_graph.nodes[node]['Radius'] = para_list[10]
        return mol_graph
