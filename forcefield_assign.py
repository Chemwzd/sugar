from .forcefield import amber_parameters, obabel_parameter_assign
from .forcefield import uff4mof_parameter_assign


def amber_assign(mol):
    para_calculator = amber_parameters.Amber99Parameter()
    atom_num = mol.get_atom_number()
    radii_i, epsilon_i, charge_i = para_calculator.get_parameters(mol)

    return [atom_num, radii_i, epsilon_i, charge_i]


def uff_assign(mol):
    atom_num = mol.get_atom_number()
    host_ob_assign = obabel_parameter_assign.OBParameterAssign(mol)
    host_graph = host_ob_assign.mol_to_graph()
    D_i = [host_graph.nodes[i]['D1'] for i in host_graph]
    x_i = [host_graph.nodes[i]['x1'] for i in host_graph]
    q_i = [host_graph.nodes[i]['partial_charge'] for i in host_graph]
    return [atom_num, D_i, x_i, q_i]


def uff4mof_assign(mol,
                   mol_path,
                   format_in='mol',
                   ff_name='UFF',
                   charge='qeq'
                   ):
    atom_num = mol.get_atom_number()
    uff4mof_assign_para = uff4mof_parameter_assign.UFF4MOFAssign(
        molecule=mol,
        mol_path=mol_path,
        format_in=format_in,
        ff_name=ff_name,
        charge=charge
    )
    host_graph = uff4mof_assign_para.get_parameter()
    D_i = [host_graph.nodes[i]['D1'] for i in host_graph]
    x_i = [host_graph.nodes[i]['x1'] for i in host_graph]
    q_i = [host_graph.nodes[i]['partial_charge'] for i in host_graph]
    return [atom_num, D_i, x_i, q_i]
