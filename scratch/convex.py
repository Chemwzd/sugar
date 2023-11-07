# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 下午4:28
# @Author  : wzd
# @File    : convex.py
from main.molecule import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

host_molecule = molecule.HostMolecule.init_from_mol_file(
    '/home/wzd/Desktop/HostGuest_for_Cages/main/mol/150_OZECAY02.mol')
convex_hull = np.array(host_molecule.convex_hull_on_mol())
print(convex_hull)

x = np.array([convex_hull[i][0] for i in range(len(convex_hull))])
y = np.array([convex_hull[i][1] for i in range(len(convex_hull))])
z = np.array([convex_hull[i][2] for i in range(len(convex_hull))])

ax = plt.subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()
# x = []
# y = []
# z = []
# with open('/home/wzd/Desktop/HostGuest_for_Cages/PoreMapper-main/examples/min_example_output/cc3_pore_70.xyz') as f:
#     content = f.readlines()
#
# for i in range(len(content)):
#     if content[i][0] == 'B':
#         x.append(float(content[i].split()[1_Energy_Contrast]))
#         y.append(float(content[i].split()[2]))
#         z.append(float(content[i].split()[3]))
# print(x)
# # print(np.array(x).reshape(-1_Energy_Contrast, 1_Energy_Contrast))
#
# positions = np.concatenate((np.array(x).reshape(-1_Energy_Contrast, 1_Energy_Contrast),
#                             np.array(y).reshape(-1_Energy_Contrast, 1_Energy_Contrast),
#                             np.array(z).reshape(-1_Energy_Contrast, 1_Energy_Contrast)), axis=1_Energy_Contrast)
# print(positions)
#
# dots_on_convexhull = positions[spatial.ConvexHull(positions, qhull_options='QJ').vertices]
# # dots_on_convexhull = positions[spatial.ConvexHull(positions).vertices]
# # x = np.array([dots_on_convexhull[i][0] for i in range(len(dots_on_convexhull))])
# # y = np.array([dots_on_convexhull[i][1_Energy_Contrast] for i in range(len(dots_on_convexhull))])
# # z = np.array([dots_on_convexhull[i][2] for i in range(len(dots_on_convexhull))])
# ax = plt.subplot(111, projection='3d')
# ax.scatter(x, y, z)
# plt.show()