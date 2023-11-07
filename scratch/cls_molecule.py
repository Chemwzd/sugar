# -*- coding: utf-8 -*-
# @Time    : 2022/7/6 下午8:19
# @Author  : wzd
# @File    : cls_molecule.py
from main import molecule

mol1 = molecule.HostMolecule.init_from_mol_file('/home/wzd/Desktop/HostGuest_for_Cages/main/mol/150_OZECAY02.mol')
print(f"windows : {mol1.cal_windows()}")
print(f"Pore volume : {mol1.cal_pore_volume_opt()}")
print(f"Pore diameter : {mol1.cal_pore_diameter_opt()}")
print(f"Max diameter of molecule : {mol1.cal_max_diameter()}")