# -*- coding: utf-8 -*-
# @Time    : 2022/9/15 下午1:44
# @Author  : wzd
# @File    : voronoi_diagram_.py
import random
import warnings

import pyvoro
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import multiprocessing as mp

import sklearn.cluster

from .utilities import cal_dis_mat, distance_between_dot, calc_rmsd, rotation_with_axis, check_interaction, norm_vector
from .molecule import HostGuestComplex, GuestMolecule
from .forcefield import potential
from .forcefield.potential import UFF4MOFPotential

from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from typing import Iterable
from scipy.optimize import minimize
from functools import partial


class VoronoiDiagram:
    """
    Use Voronoi diagram and BFGS algorithm to find the minimum energy
    host-guest conformations.

    """

    def __init__(
            self,
            molecule,
            limit: Iterable = (0.1, 0.1, 0.1),
            radii_cutoff: float = 3.0,
            block_size: float = 0.5,
            periodic=None,
            remove_h=False,
            rotation_step_size: float = 1,
            translation_step_size: int = 1
    ):
        """
        Generate the voronoi diagram of the molecule.

        Note
        ---------------------
        If you meet the error 'pyvoro.voroplusplus.VoronoiPlusPlusError: number of cells found was not equal to the number of particles',
        please try to increase the size of limit, such as limit=(0.5, 0.5, 0.5).

        Parameters
        ---------------------
        molecule: 'HostMolecule'
                The molecule to generate voronoi diagram.

        limit: 'list' or 'ndarray'
                The value to be added for the boundary of the box.
                ---------------------------------------------------------------------------------------------
                Example:
                    limit = [0.5, 0.5, 0.5]
                    It means that the boundary of the box will be extended by 0.5 Angstrom in each direction.
                ---------------------------------------------------------------------------------------------

        radii_cutoff: 'float'
                The radii cutoff for the voronoi vertices.
                If r_vertex < radii_cutoff, the vertex will be removed.

        block_size: 'float'
                Max distance between two points that might be adjacent (sets
            voro++ block sizes.)

        periodic: 2-list of bools indicating x and y periodicity of the system box.
                Default to be None.

        remove_h: 'bool'
                Whether to remove H atoms before calculating voronoi diagram.
                Default to be 'False'.

        """
        molecule = molecule.translate_to_new_origin([0, 0, 0])
        self._min_energy = 99999999.99
        self._limit = limit
        self._molecule = molecule
        self._atom_positions = molecule.get_positions()
        self._atom_radii = [i.get_atomic_radius() for i in molecule.get_atoms()]
        self._block_size = block_size
        self._centroid = molecule.get_centroid_remove_h()
        self._radii_cutoff = radii_cutoff
        self._remove_h = remove_h
        self._translation_step_size = translation_step_size
        self._rotation_step_size = rotation_step_size
        if periodic is None:
            self._periodic = [False] * 3
        else:
            self._periodic = periodic
        self._radii_remove_h, self._positions_remove_h = self._mol_remove_h()
        self._vertices, self._adjacency = self._generate_voro_info()

    def _mol_remove_h(self):
        atoms = [i for i in self._molecule.get_atoms()]
        radii = [i.get_atomic_radius() for i in atoms if i.get_element() != 'H']
        positions = self._molecule.get_positions_remove_h()
        return radii, positions

    def _generate_voro_info(
            self
    ):
        """
        Using pyvoro to generate the information of the voronoi diagram.
        See pyvoro for more details.

        Reference:
        ---------------------
        pyvoro: https://github.com/joe-jordan/pyvoro

        Returns
        ---------------------
        vertices: 'list'
            The vertices of the voronoi diagram.
            In the form of [[[x1, y1, z1], [x2, y2, z2], ...], ...]

        adjacency: 'list'
            The adjacency of every vertex.
            In the form of [[[1_Energy_Contrast, 2, 3], [3, 6, 7]...]], ...]

        """
        if self._remove_h:
            positions = self._positions_remove_h
            radii = self._radii_remove_h
        else:
            positions = self._atom_positions
            radii = self._atom_radii
        msg = pyvoro.compute_voronoi(
            positions,
            [
                [np.min(self._atom_positions[:, 0] - self._limit[0]),
                 np.max(self._atom_positions[:, 0] + self._limit[0])],
                [np.min(self._atom_positions[:, 1] - self._limit[1]),
                 np.max(self._atom_positions[:, 1] + self._limit[1])],
                [np.min(self._atom_positions[:, 2] - self._limit[2]),
                 np.max(self._atom_positions[:, 2] + self._limit[2])]
            ],
            self._block_size,
            radii=radii,
            periodic=self._periodic
        )

        vertices = [np.around(msg[i]['vertices'], 3) for i in range(len(msg))]
        adjacency = [msg[i]['adjacency'] for i in range(len(msg))]
        return vertices, adjacency

    def plot_voro(
            self,
            save_path: str = None,
            show_atoms=False,
            dot_color='dodgerblue',
            dot_shape='o',
            dot_size=10,
            dot_alpha=0.6,
            linestyle='--',
            linecolor='lightskyblue',
            linewidth=0.3,
            line_alpha=0.3,
            dpi=800
    ):
        """
        Plot the voronoi diagram of the molecule.

        Note
        ---------------------
        Read the doc of pyvoro for more details.

        Parameters
        ---------------------
        save_path: 'str'
            Path to save the figure.

        show_atoms: 'bool'
            Whether to show atoms or not.

        dot_color='dodgerblue',

        dot_shape='o',

        dot_alpha=0.6,

        linestyle='--',

        linecolor='lightskyblue',

        linewidth=0.3,

        line_alpha=0.3

        """
        if self._remove_h:
            positions = self._positions_remove_h
            radii = self._radii_remove_h
        else:
            positions = self._atom_positions
            radii = self._atom_radii
        msg = pyvoro.compute_voronoi(
            positions,
            [
                [np.min(self._atom_positions[:, 0] - self._limit[0]),
                 np.max(self._atom_positions[:, 0] + self._limit[0])],
                [np.min(self._atom_positions[:, 1] - self._limit[1]),
                 np.max(self._atom_positions[:, 1] + self._limit[1])],
                [np.min(self._atom_positions[:, 2] - self._limit[2]),
                 np.max(self._atom_positions[:, 2] + self._limit[2])]
            ],
            self._block_size,
            radii=radii,
            periodic=self._periodic
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(msg)):
            vertices_position = np.around(msg[i]['vertices'], 3)
            adjacency = msg[i]['adjacency']
            ax.scatter([x[0] for x in vertices_position], [x[1] for x in vertices_position],
                       [x[2] for x in vertices_position], c=dot_color, marker=dot_shape, linestyle=linestyle,
                       alpha=dot_alpha, s=dot_size)
            for j in range(len(adjacency)):
                for k in range(3):
                    ax.plot([vertices_position[j][0], vertices_position[adjacency[j][k]][0]],
                            [vertices_position[j][1], vertices_position[adjacency[j][k]][1]],
                            [vertices_position[j][2], vertices_position[adjacency[j][k]][2]], c=linecolor,
                            linewidth=linewidth,
                            linestyle=linestyle, alpha=line_alpha)
        if show_atoms:
            for l in range(len(self._atom_positions)):
                ax.scatter(self._atom_positions[l][0], self._atom_positions[l][1], self._atom_positions[l][2], c='g',
                           marker='o', s=self._atom_radii[l] * 100)
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi)
        plt.show()
        return None

    def remove_vertices(
            self,
            method=KMeans,
            verbose=False,
            **kwargs
    ):
        """
        Remove the voronoi vertices that are too close to the atoms,
        and the vertices that are too far to the host centroid.

        Note
        ---------------------
        In voronoi algorithm, the vertices are generated around the atoms.
        If calculate all the vertices, it will take a lot of time and inefficient.
        So we need to remove the vertices that are too close to the atoms.
        Firstly, we calculate the distance between each vertex and the atoms.
        Secondly, we assign a radii for each vertex, which is the minimum distance
            between the vertex and the atoms.
        Thirdly, we remove the vertices that radii is smaller than the 'cutoff' or
            bigger than the radii of host molecule.

        Parameters
        ---------------------
        method: 'class'
            The cluster method based on sklearn.
        verbose: 'bool'
            Default to be False, print the information of the vertices.
        *args: 'tuple'
            The parameters of the method.

        Returns
        ----------------------
        final_vertices: 'list'
            The final vertices after removing process.
            In the form of [[x1, y1, z1], [x2, y2, z2], ...]

        """
        vertices_radii = []
        centroid_distance = []
        if self._remove_h:
            if len(self._positions_remove_h) != len(self._vertices):
                raise ValueError("The number of vertices and atoms must be equal")
        else:
            if len(self._atom_positions) != len(self._vertices):
                raise ValueError("The number of vertices and atoms must be equal")

        # Calculate the minimum distance between the vertices and the atoms,
        # and the distance between the vertices and the host centroid.
        for i in range(len(self._vertices)):
            radii_list = [round(np.min(cal_dis_mat(j.reshape((1, 3)), self._atom_positions)), 4) for j in
                          self._vertices[i]]
            centroid_dis_list = [round(distance_between_dot(self._centroid.reshape((1, 3)), j.reshape((1, 3))), 4) for j
                                 in self._vertices[i]]
            vertices_radii.append(radii_list)
            centroid_distance.append(centroid_dis_list)

        new_vertices_positions = []
        new_vertices_distance = []

        for i in range(len(self._vertices)):
            for j in range(len(self._vertices[i])):
                if vertices_radii[i][j] >= self._radii_cutoff and centroid_distance[i][j] \
                        < self._molecule.cal_pore_diameter() / 2:
                    new_vertices_positions.append(self._vertices[i][j])
                    new_vertices_distance.append(vertices_radii[i][j])

        try:
            if method == AgglomerativeClustering:
                # AgglomerativeClustering is a hierarchical clustering method.
                # Sometimes it will generate too many clusters, so we need to
                # cluster the vertices twice.
                Cluster_Method = method(**kwargs)
                cluster = Cluster_Method.fit(new_vertices_positions)
                labels = cluster.labels_

                if verbose:
                    print(f"Number of clusters: {len(np.unique(labels))}")
                    print(f"Labels : {labels}\n")

                # Link the cluster.labels_ to new_vertices_positions
                # and return the vertex which is the farthest to the host atoms.
                cluster_vertices = []
                cluster_vertices_distance = []
                for i in range(len(np.unique(labels))):
                    index = np.where(labels == i)
                    cluster_vertices.append(
                        new_vertices_positions[
                            index[0][
                                np.argmax(
                                    np.array(new_vertices_distance)[index]
                                )
                            ]
                        ]
                    )
                    cluster_vertices_distance.append(
                        new_vertices_distance[
                            index[0][
                                np.argmax(
                                    np.array(new_vertices_distance)[index]
                                )
                            ]
                        ]
                    )

                # Use Hierarchy cluster twice to decrease the number of vertices.
                if len(cluster_vertices) > 2:
                    cluster_vertices2 = []
                    cluster2 = Cluster_Method.fit(cluster_vertices)
                    labels2 = cluster2.labels_
                    for i in range(len(np.unique(labels2))):
                        index2 = np.where(labels2 == i)
                        cluster_vertices2.append(
                            cluster_vertices[
                                index2[0][
                                    np.argmax(
                                        np.array(cluster_vertices_distance)[index2]
                                    )
                                ]
                            ]
                        )
                    return np.array(cluster_vertices2)
                return np.array(cluster_vertices)
            else:
                Cluster_Method = method(**kwargs)
                cluster = Cluster_Method.fit(new_vertices_positions)
                labels = cluster.labels_

                if verbose:
                    print(f"Number of clusters: {len(np.unique(labels))}")
                    print(f"Labels : {labels}\n")
                    # print(np.array(new_vertices_positions).tolist())

                # Link the cluster.labels_ to new_vertices_positions
                # and return the vertex which is the farthest to the host atoms.
                cluster_vertices = []
                for i in range(len(np.unique(labels))):
                    index = np.where(labels == i)
                    cluster_vertices.append(
                        new_vertices_positions[
                            index[0][
                                np.argmax(
                                    np.array(new_vertices_distance)[index]
                                )
                            ]
                        ]
                    )

                return np.array(cluster_vertices)
        except Exception as e:
            warnings.warn(f"Clustering failed, return all vertices. Error: {e}")
            return np.array(new_vertices_positions)

    def get_binding_positions(self, method=KMeans,
                              verbose=False,
                              **kwargs):
        """
        Get final voronoi sites.

        Parameters:
        ------------------------------
        remove_h: 'bool'
            Whether to remove H atoms before calculating voronoi diagram.
            Default to be 'True'.

        """
        return np.round(self.remove_vertices(method, verbose, **kwargs), 3)

    def binding(
            self,
            binding_node,
            guest,
            host_par,
            guest_par,
            calculator=potential.AmberPotential,
            random_seed=1000,
            times=20,
            verbose=False
    ):
        """
        Dock guest molecule with BFGS algorithm.

        Notes
        ------------------------------
        This function performs only BFGS algorithm at the specific 'binding_node'.
        If you want to dock guest molecule on every voronoi node, please use function 'voro_binding'.

        Parameters
        ------------------------------
        binding_node: 'np.ndarray'
                The binding positions of the guest molecule.

        guest: 'GuestMolecule'
                The guest molecule to be docked in the host molecule.

        calculator: The potential function to be used, such as 'potential.AmberPotential'.
                Default to be 'potential.AmberPotential'.

        random_seed: 'int'
                Random seed to be applied on 'np.random.seed()' and 'random.seed()'
                Default to be 1000.

        times: 'int'
                Number of iterations in BFGS algorithm.
                Default to be 100.

        verbose: 'bool'
                Whether to print the information of the binding process.
                Default to be False.

        mol_path: 'str'
                The path of the molecule which is needed to be corrected
                before calculation.

        Returns
        ------------------------------
        'HostGuestComplex'
                The final conformation to be generated through BFGS algorithm.

        """

        def get_rotated_energy(rot_angle, rot_axis, host, guest, host_par, guest_par):
            new_guest = guest.rotate_guest_molecule(rot_axis, rot_angle)
            if calculator == UFF4MOFPotential or potential.AmberPotential or potential.UFFPotential:
                energy = calculator(host, new_guest, host_par, guest_par).cal_potential()
            else:
                energy = calculator(host, new_guest).cal_potential()
            return energy

        np.random.seed(random_seed)
        random.seed(random_seed)

        binding_node = np.array(binding_node).reshape((1, 3))
        guest = guest.translate_to_new_origin(binding_node)
        host = self._molecule
        energy = []
        rot_angles = random.random() * 2 - 1 * self._rotation_step_size

        final_energy = 0
        final_energy_list = []

        for i in range(times):
            theta = 2 * np.pi * random.random()
            phi = np.arccos(2 * random.random() - 1)
            rotation_axis = np.array([np.sin(phi) * np.cos(theta),
                                      np.sin(phi) * np.sin(theta),
                                      np.cos(phi)])

            result = minimize(
                get_rotated_energy,
                x0=rot_angles,
                args=(rotation_axis, host, guest, host_par, guest_par),
                method='L-BFGS-B',
                tol=0.001
            )
            energy1 = result.fun
            energy.append(energy1)

            if energy1 < self._min_energy:
                self._min_energy = energy1
                rot_angles = result.x[0]
                guest = guest.rotate_guest_molecule(rotation_axis, rot_angles)
                final_energy = energy1
                final_energy_list.append(final_energy)

        if verbose:
            print(f"energy during binding : {energy}")
            print(f"opt-energy : {final_energy_list}\n")
        if final_energy == 0:
            return None, final_energy
        else:
            return HostGuestComplex.init_from_molecule(host, guest), final_energy

    def binding_unfix(
            self,
            binding_node,
            guest,
            host_par,
            guest_par,
            calculator=potential.AmberPotential,
            random_seed=1000,
            times=20,
            verbose=False
    ):
        """
        Dock guest molecule with BFGS algorithm.

        Notes
        ------------------------------
        This function performs only BFGS algorithm at the specific 'binding_node'.
        If you want to dock guest molecule on every voronoi node, please use function 'voro_binding'.

        Parameters
        ------------------------------
        binding_node: 'np.ndarray'
                The binding positions of the guest molecule.

        guest: 'GuestMolecule'
                The guest molecule to be docked in the host molecule.

        calculator: The potential function to be used, such as 'potential.AmberPotential'.
                Default to be 'potential.AmberPotential'.

        random_seed: 'int'
                Random seed to be applied on 'np.random.seed()' and 'random.seed()'
                Default to be 1000.

        times: 'int'
                Number of iterations in BFGS algorithm.
                Default to be 100.

        verbose: 'bool'
                Whether to print the information of the binding process.
                Default to be False.

        Returns
        ------------------------------
        'HostGuestComplex'
                The final conformation to be generated through BFGS algorithm.

        """

        def get_rotated_energy_unfix(vec, rot_axis, host, guest, host_par, guest_par):
            new_guest = guest.rotate_guest_molecule(rot_axis, vec[0]).move_guest_molecule([vec[1], vec[2], vec[3]])
            if calculator == UFF4MOFPotential or potential.AmberPotential or potential.UFFPotential:
                energy = calculator(host, new_guest, host_par, guest_par).cal_potential()
            else:
                energy = calculator(host, new_guest).cal_potential()
            return energy

        np.random.seed(random_seed)
        random.seed(random_seed)

        binding_node = np.array(binding_node).reshape((1, 3))
        guest = guest.translate_to_new_origin(binding_node)
        host = self._molecule
        energy = []
        rot_angles = random.random() * 2 - 1 * self._rotation_step_size
        # Get the random vector.
        rand_vector = norm_vector(np.random.rand(3) * 2 - 1)
        # Get the translation vector.
        translation_vector = rand_vector * self._translation_step_size

        final_energy = 0

        for i in range(times):
            theta = 2 * np.pi * random.random()
            phi = np.arccos(2 * random.random() - 1)
            rotation_axis = np.array([np.sin(phi) * np.cos(theta),
                                      np.sin(phi) * np.sin(theta),
                                      np.cos(phi)])

            x0 = [rot_angles, translation_vector[0], translation_vector[1], translation_vector[2]]

            result = minimize(
                get_rotated_energy_unfix,
                x0=x0,
                args=(rotation_axis, host, guest, host_par, guest_par),
                method='L-BFGS-B',
                tol=0.001
            )
            energy1 = result.fun
            energy.append(energy1)

            if energy1 < self._min_energy:
                self._min_energy = energy1
                x0 = [result.x[0], result.x[1], result.x[2], result.x[3]]
                guest = guest.rotate_guest_molecule(rotation_axis, x0[0]).move_guest_molecule(
                    [x0[1], x0[2], x0[3]])
                final_energy = energy1

        if verbose:
            print(x0)
            print(f"energy during binding : {energy}\n")
        if final_energy == 0:
            return None, final_energy
        else:
            return HostGuestComplex.init_from_molecule(host, guest), final_energy

    def voro_binding(
            self,
            voronoi_nodes,
            guest,
            host_par,
            guest_par,
            calculator=potential.AmberPotential,
            random_seed=1000,
            times=20,
            verbose=False,
            fix=True
    ):
        """
        binding guest molecule on each voronoi node which is generated by the
        function 'voronoi.binding'.

        Notes
        --------------------------------
        It's recommended to use function 'run_binding' instead of this function.
        Because this function uses only 1_Energy_Contrast thread to perform binding process.

        Parameters
        --------------------------------
        voronoi_nodes: 'np.ndarray'
                The binding positions of the guest molecule.

        guest: 'GuestMolecule'
                The guest molecule to be docked in the host molecule.

        calculator: The potential function to be used, such as 'potential.AmberPotential'.
                Default to be 'potential.AmberPotential'.

        random_seed: 'int'
                Random seed to be applied on 'np.random.seed()' and 'random.seed()'
                Default to be 1000.

        times: 'int'
                Number of iterations in BFGS algorithm.
                Default to be 100.

        verbose: 'bool'
                Whether to print the information of the binding process.
                Default to be False.

        mol_path: 'str'
                The path of the molecule which is needed to be corrected
                before calculation.

        """
        if fix:
            conformers = [self.binding(node,
                                       guest=guest,
                                       host_par=host_par,
                                       guest_par=guest_par,
                                       calculator=calculator,
                                       random_seed=random_seed,
                                       times=times,
                                       verbose=verbose,
                                       fix=fix) for node in voronoi_nodes]
        else:
            conformers = [self.binding_unfix(node,
                                             guest=guest,
                                             host_par=host_par,
                                             guest_par=guest_par,
                                             calculator=calculator,
                                             random_seed=random_seed,
                                             times=times,
                                             verbose=verbose,
                                             fix=fix) for node in voronoi_nodes]
        return conformers

    def _run_binding(
            self,
            vertex,
            guest,
            host_par,
            guest_par,
            calculator,
            random_seed,
            times,
            verbose
    ):
        binding = self.binding(
            binding_node=vertex,
            guest=guest,
            host_par=host_par,
            guest_par=guest_par,
            calculator=calculator,
            random_seed=random_seed,
            times=times,
            verbose=verbose
        )
        return binding

    def _run_binding_unfix(
            self,
            vertex,
            guest,
            host_par,
            guest_par,
            calculator,
            random_seed,
            times,
            verbose
    ):
        binding = self.binding_unfix(
            binding_node=vertex,
            guest=guest,
            host_par=host_par,
            guest_par=guest_par,
            calculator=calculator,
            random_seed=random_seed,
            times=times,
            verbose=verbose
        )

        return binding

    def run_binding(
            self,
            voronoi_nodes,
            guest,
            host_par,
            guest_par,
            calculator=potential.AmberPotential,
            random_seed=1000,
            times=20,
            verbose=False,
            fix=True,
            return_all=True
    ):
        """
        ===================================================================
        ==                  Maybe replaced by 'binding_mp'               ==
        ==       Add multiprocessing to calculate every BFGS loop        ==
        ===================================================================
        Running binding code in parallel.

        Parameters
        --------------------------------
        voronoi_nodes: 'np.ndarray'
                The binding positions of the guest molecule.

        guest: 'GuestMolecule'
                The guest molecule to be docked in the host molecule.

        calculator: The potential function to be used, such as 'potential.AmberPotential'.
                Default to be 'potential.AmberPotential'.

        random_seed: 'int'
                Random seed to be applied on 'np.random.seed()' and 'random.seed()'
                Default to be 1000.

        times: 'int'
                Number of iterations in BFGS algorithm.
                Default to be 100.

        verbose: 'bool'
                Whether to print the information of the binding process.
                Default to be False.

        fix: 'bool'

        mol_path: 'str'
                The path of the molecule which is needed to be corrected
                before calculation.

        return_all: 'bool'
                Whether to return all the conformers or only the lowest energy conformer.
                Default to be True.

        Returns
        ----------------------
        In the form of:
            [{task1: (HostGuestComplex_object1, final_energy1)},
            {task2: (HostGuestComplex_object2, final_energy2)}, ...]

        After Running 'run_binding', you can write these conformations into .mol file.

        """

        num_cores = int(mp.cpu_count())
        pool = mp.Pool(num_cores)

        if fix:
            print(f'---------Fix is {fix}, Guest is fixed on the predicted nodes.--------\n')
            results = [pool.apply_async(
                self._run_binding,
                args=(vertex, guest, host_par, guest_par, calculator, random_seed, times, verbose)
            ) for vertex in voronoi_nodes]
            output = [p.get() for p in results]

            pool.close()
            pool.join()
            if return_all:
                return output
            else:
                for i in output:
                    if i[1] == min([j[1] for j in output]):
                        return i
        else:
            print(f'---------Fix is {fix}, Guest can rotate and translate simultaneously.--------\n')
            results = [pool.apply_async(
                self._run_binding_unfix,
                args=(vertex, guest, host_par, guest_par, calculator, random_seed, times, verbose)
            ) for vertex in voronoi_nodes]
            output = [p.get() for p in results]

            pool.close()
            pool.join()
            if return_all:
                return output
            else:
                for i in output:
                    if i[1] == min([j[1] for j in output]):
                        return i

    def run_binding_with_confs(
            self,
            voronoi_nodes,
            guest,
            host_par,
            guest_par,
            calculator=potential.AmberPotential,
            random_seed=1000,
            times=20,
            verbose=False,
            num_guest_confs=None,
            fix=True,
            return_all=True
    ):
        """
        ===================================================================
        ==                  Maybe replaced by 'binding_mp'               ==
        ==       Add multiprocessing to calculate every BFGS loop        ==
        ===================================================================
        Running binding code in parallel.

        Parameters
        --------------------------------
        voronoi_nodes: 'np.ndarray'
                The binding positions of the guest molecule.

        guest: 'GuestMolecule'
                The guest molecule to be docked in the host molecule.

        host_par: 'list'
                The force field parameters of the host molecule.

        guest_par: 'list'
                The force field parameters of the guest molecule.

        calculator: The potential function to be used, such as 'potential.AmberPotential'.
                Default to be 'potential.AmberPotential'.

        random_seed: 'int'
                Random seed to be applied on 'np.random.seed()' and 'random.seed()'
                Default to be 1000.

        times: 'int'
                Number of iterations in BFGS algorithm.
                Default to be 100.

        verbose: 'bool'
                Whether to print the information of the binding process.
                Default to be False.

        num_guest_confs: 'int'
                Number of guest molecule conformers.
                Default to be 'None'.

        fix: 'bool'
                Whether to fix the guest molecule on the predicted nodes.

        return_all: 'bool'
                Whether to return all the conformations.
                Default to be 'True'.

        Returns
        ----------------------
        In the form of:
            [{task1: (HostGuestComplex_object1, final_energy1)},
            {task2: (HostGuestComplex_object2, final_energy2)}, ...]

        After Running 'run_binding', you can write these conformations into .mol file.

        """
        num_cores = int(mp.cpu_count())
        pool = mp.Pool(num_cores)
        com = guest.get_centroid_remove_h()
        conf_list = [conf.translate_to_new_origin(com) for conf in guest.mol_to_conformers(guest, num_guest_confs)]
        # conf_list.append(guest)
        if fix:
            # print(f'---------Fix is {fix}, Guest is fixed on the predicted nodes.--------\n')
            results = [pool.apply_async(
                self._run_binding,
                args=(vertex, conf, host_par, guest_par, calculator, random_seed, times, verbose)
            ) for vertex in voronoi_nodes for conf in conf_list]
            output = [p.get() for p in results]

            pool.close()
            pool.join()
            if return_all:
                return output
            else:
                for i in output:
                    if i[1] == min([j[1] for j in output]):
                        return i

        else:
            # print(f'---------Fix is {fix}, Guest can rotate and translate simultaneously.--------\n')
            results = [pool.apply_async(
                self._run_binding_unfix,
                args=(vertex, conf, host_par, guest_par, calculator, random_seed, times, verbose)
            ) for vertex in voronoi_nodes for conf in conf_list]
            output = [p.get() for p in results]

            pool.close()
            pool.join()
            if return_all:
                return output
            else:
                for i in output:
                    if i[1] == min([j[1] for j in output]):
                        return i

    def run_binding_with_confs_v2(
            self,
            voronoi_nodes,
            guest,
            host_par,
            guest_par,
            calculator=potential.AmberPotential,
            random_seed=1000,
            times=20,
            verbose=False,
            num_guest_confs=None,
            fix=True,
            return_all=True
    ):
        """
        ===================================================================
        ==                  Maybe replaced by 'binding_mp'               ==
        ==       Add multiprocessing to calculate every BFGS loop        ==
        ===================================================================
        Running binding code in parallel.

        Parameters
        --------------------------------
        voronoi_nodes: 'np.ndarray'
                The binding positions of the guest molecule.

        guest: 'GuestMolecule'
                The guest molecule to be docked in the host molecule.

        host_par: 'list'
                The force field parameters of the host molecule.

        guest_par: 'list'
                The force field parameters of the guest molecule.

        calculator: The potential function to be used, such as 'potential.AmberPotential'.
                Default to be 'potential.AmberPotential'.

        random_seed: 'int'
                Random seed to be applied on 'np.random.seed()' and 'random.seed()'
                Default to be 1000.

        times: 'int'
                Number of iterations in BFGS algorithm.
                Default to be 100.

        verbose: 'bool'
                Whether to print the information of the binding process.
                Default to be False.

        num_guest_confs: 'int'
                Number of guest molecule conformers.
                Default to be 'None'.

        fix: 'bool'
                Whether to fix the guest molecule on the predicted nodes.

        return_all: 'bool'
                Whether to return all the conformations.
                Default to be 'True'.

        Returns
        ----------------------
        In the form of:
            [{task1: (HostGuestComplex_object1, final_energy1)},
            {task2: (HostGuestComplex_object2, final_energy2)}, ...]

        After Running 'run_binding', you can write these conformations into .mol file.

        """
        num_cores = int(mp.cpu_count())
        pool = mp.Pool(num_cores)
        com = guest.get_centroid_remove_h()
        conf_list = [conf.translate_to_new_origin(com) for conf in guest.mol_to_conformers(guest, num_guest_confs)]
        # conf_list.append(guest)
        if fix:
            # print(f'---------Fix is {fix}, Guest is fixed on the predicted nodes.--------\n')
            results = [pool.apply_async(
                self._run_binding,
                args=(vertex, conf, host_par, guest_par, calculator, random_seed, times, verbose)
            ) for conf in conf_list for vertex in voronoi_nodes]
            output = [p.get() for p in results]

            pool.close()
            pool.join()
            if return_all:
                return output
            else:
                for i in output:
                    if i[1] == min([j[1] for j in output]):
                        return i

        else:
            # print(f'---------Fix is {fix}, Guest can rotate and translate simultaneously.--------\n')
            results = [pool.apply_async(
                self._run_binding_unfix,
                args=(vertex, conf, host_par, guest_par, calculator, random_seed, times, verbose)
            ) for vertex in voronoi_nodes for conf in conf_list]
            output = [p.get() for p in results]

            pool.close()
            pool.join()
            if return_all:
                return output
            else:
                for i in output:
                    if i[1] == min([j[1] for j in output]):
                        return i

    def _check_shape(
            self,
            binding_node,
            guest,
            cutoff=1.15,
            step_size=20
    ):
        """
        Check whether the guest molecule is suitable for rotating 360 degrees on the binding node.

        Returns: list
                [[float, float, float], ...]

        """
        total_rates = []
        total_complex = []
        total_guest = []

        for node in binding_node:
            guest = guest.translate_to_new_origin(node)
            centroid = guest.get_centroid_remove_h()
            axis = np.array(
                [[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
            )

            rates = []
            complex_list = []
            guest_list = []

            for i in range(3):
                single_interactions = []

                # single direction rotation list.
                direction_interactions = []
                guests = []

                for angle in range(0, 360, step_size):
                    # Translate the points to new coordinate system.
                    guest_trans = guest.move_guest_molecule(movement=-centroid)
                    guest_rotated = guest_trans.rotate_guest_molecule(axis=axis[i], angle=angle * np.pi / 180)
                    guest_trans = guest_rotated.move_guest_molecule(movement=centroid)
                    interaction = HostGuestComplex.check_interaction(
                        self._atom_positions,
                        guest_trans.get_positions(),
                        self._molecule.get_atoms(),
                        guest_trans.get_atoms(),
                        cutoff=cutoff
                    )
                    direction_interactions.append(HostGuestComplex.init_from_molecule(self._molecule, guest_trans))
                    guests.append(guest_trans)
                    # complex_list.append(HostGuestComplex.init_from_molecule(self._molecule, guest_trans))
                    if len(interaction) > 0:
                        single_interactions.append(interaction)

                # give the rate of interaction
                rates.append(len(single_interactions) / (360 / step_size))
                complex_list.append(direction_interactions)
                guest_list.append(guests)
            total_rates.append(rates)
            total_complex.append(complex_list)
            total_guest.append(guest_list)
        return total_rates, total_complex, total_guest

    def check_shape(self,
                    binding_node,
                    guest,
                    cutoff=1.15,
                    step_size=2,
                    min_rate_cutoff=0.2,
                    max_rate_cutoff=0.8
                    ):
        """


        """
        total_rates, _, _ = self._check_shape(binding_node, guest, cutoff, step_size)
        print(total_rates)
        proper_nodes = []
        for n in range(len(binding_node)):
            #
            hit = 0
            flag = True
            for r in range(len(total_rates[n])):
                direction = ['x', 'y', 'z']
                # lower rate means smaller space that the guest can easily escape the host cavity.
                if total_rates[n][r] < min_rate_cutoff:
                    print(f"Guest is too small to dock on {direction[r]} of the node: {binding_node[n]}")
                    flag = False
                    break

                # larger rate means larger space that might be unsuitable for host cavity.
                elif total_rates[n][r] > max_rate_cutoff:
                    hit += 1
            if hit < 3 and flag:
                proper_nodes.append(binding_node[n])
            elif hit == 3:
                print(f"Guest is too large to dock on {direction[r]} of the node: {binding_node[n]}")

        return proper_nodes

    def binding_mp(
            self,
            binding_node,
            guest,
            calculator=potential.AmberPotential,
            random_seed=1000,
            times=100,
            verbose=False,
            mol_path=None
    ):
        """
        ===================================================================
                                UNDER DEVELOPMENT
        ===================================================================
        ==                         Developing...                         ==
        ==       Add multiprocessing to calculate every BFGS loop        ==
        ===================================================================

        Dock guest molecule with BFGs algorithm.

        Notes
        ------------------------------
        This function uses new guest that has been optimized by BFGS algorithm as the initial
        guest in every for loop.

        Parameters
        ------------------------------
        binding_node: 'np.ndarray'
                The binding positions of the guest molecule.

        calculator: The potential function to be used, such as 'potential.AmberPotential'.
                Default to be 'potential.AmberPotential'.

        random_seed: 'int'
                Random seed to be applied on 'np.random.seed()' and 'random.seed()'
                Default to be 1000.

        times: 'int'
                Number of iterations in BFGS algorithm.
                Default to be 100.

        verbose: 'bool'
                Whether to print the information of the binding process.
                Default to be False.

        mol_path: 'str'
                The path of the molecule which is needed to be corrected
                before calculation.

        Returns
        ------------------------------
        'HostGuestComplex'
                The final conformation to be generated through BFGs algorithm.

        """

        def get_rotated_energy(rot_angle, rot_axis, host, guest):
            new_guest = guest.rotate_guest_molecule(rot_axis, rot_angle)
            if calculator == UFF4MOFPotential:
                energy = calculator(host, new_guest).cal_potential(mol_path=mol_path, format_in='mol')
            else:
                energy = calculator(host, new_guest).cal_potential()
            return energy

        np.random.seed(random_seed)
        random.seed(random_seed)

        binding_node = np.array(binding_node).reshape((1, 3))
        guest = guest.translate_to_new_origin(binding_node)
        host = self._molecule
        energy = {}

        num_cores = int(mp.cpu_count())

        def run_bfgs(name, guest):
            rot_angles = random.random() * 2 - 1

            theta = 2 * np.pi * random.random()
            phi = np.arccos(2 * random.random() - 1)
            rotation_axis = np.array([np.sin(phi) * np.cos(theta),
                                      np.sin(phi) * np.sin(theta),
                                      np.cos(phi)])

            result = minimize(
                get_rotated_energy,
                x0=rot_angles,
                args=(rotation_axis, host, guest),
                method='L-BFGS-B',
                tol=0.001
            )
            energy1 = result.fun
            guest = guest.rotate_guest_molecule(rotation_axis, result.x[0])
            complex1 = HostGuestComplex.init_from_molecule(host, guest)
            energy[complex1] = energy1
            return {name: {complex1: energy1}}

        pool = mp.Pool(num_cores)
        results = [pool.apply_async(run_bfgs, args=(i, guest)) for i in range(times)]
        output = [p.get() for p in results]

        if verbose:
            print(f"------binding on the node : {binding_node} has been finished.------")
            print(f"energy during binding : {energy}\n")

        return output

    def voro_binding_mp(
            self,
            voronoi_nodes,
            calculator=potential.AmberPotential,
            random_seed=1000,
            times=30,
            verbose=False
    ):
        """
        binding guest molecule on each voronoi node which is generated by the
        function 'voronoi.binding_opt()'.

        Parameters
        --------------------------------
        voronoi_nodes: 'np.ndarray'
                The binding positions of the guest molecule.

        calculator: The potential function to be used, such as 'potential.AmberPotential'.
                Default to be 'potential.AmberPotential'.

        random_seed: 'int'
                Random seed to be applied on 'np.random.seed()' and 'random.seed()'
                Default to be 1000.

        times: 'int'
                Number of iterations in BFGs algorithm.
                Default to be 100.

        verbose: 'bool'
                Whether to print the information of the binding process.
                Default to be False.

        mol_path: 'str'
                The path of the molecule which is needed to be corrected
                before calculation.

        """
        conformers = [self.binding_mp(node,
                                      calculator=calculator,
                                      random_seed=random_seed,
                                      times=times,
                                      verbose=verbose) for node in voronoi_nodes]
        return conformers

    def remove_vertices_copy(
            self,
            final_vertices_threshold: int = 3,
            verbose=False
    ):
        """
        =======================================================================
        COPY AT 2023.04.20 20：16
        原始删除顶点方法（加入二次聚类，但易删除）
        ============================================================================
        Remove the voronoi vertices that are too close to the atoms,
        and the vertices that are too far to the host centroid.

        Note
        ---------------------
        In voronoi algorithm, the vertices are generated around the atoms.
        If calculate all the vertices, it will take a lot of time and inefficient.
        So we need to remove the vertices that are too close to the atoms.
        Firstly, we calculate the distance between each vertex and the atoms.
        Secondly, we assign a radii for each vertex, which is the minimum distance
            between the vertex and the atoms.
        Thirdly, we remove the vertices that radii is smaller than the 'cutoff' or
            bigger than the radii of host molecule.

        Parameters
        ---------------------
        final_vertices_threshold: 'int'
            The threshold number of the final vertices.
            Default to be 3.
            The final vertices number should be smaller than this threshold.

        verbose: 'bool'
            Default to be False, print the information of the vertices.

        Returns
        ----------------------
        final_vertices: 'list'
            The final vertices after removing process.
            In the form of [[x1, y1, z1], [x2, y2, z2], ...]

        """
        vertices_radii = []
        centroid_distance = []
        if self._remove_h:
            if len(self._positions_remove_h) != len(self._vertices):
                raise ValueError("The number of vertices and atoms must be equal")
        else:
            if len(self._atom_positions) != len(self._vertices):
                raise ValueError("The number of vertices and atoms must be equal")

        # Calculate the minimum distance between the vertices and the atoms,
        # and the distance between the vertices and the host centroid.
        for i in range(len(self._vertices)):
            radii_list = [round(np.min(cal_dis_mat(j.reshape((1, 3)), self._atom_positions)), 4) for j in
                          self._vertices[i]]
            centroid_dis_list = [round(distance_between_dot(self._centroid.reshape((1, 3)), j.reshape((1, 3))), 4) for j
                                 in self._vertices[i]]
            vertices_radii.append(radii_list)
            centroid_distance.append(centroid_dis_list)

        new_vertices_positions = []
        new_vertices_distance = []
        for i in range(len(self._vertices)):
            for j in range(len(self._vertices[i])):
                if vertices_radii[i][j] >= self._radii_cutoff and centroid_distance[i][j] \
                        < self._molecule.cal_pore_diameter() / 2:
                    new_vertices_positions.append(self._vertices[i][j])
                    new_vertices_distance.append(vertices_radii[i][j])
                # if vertices_radii[i][j] >= self._radii_cutoff:
                #     new_vertices_positions.append(self._vertices[i][j])
                #     new_vertices_distance.append(vertices_radii[i][j])

        # Add for plot the dots after screening.
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xticks([-0.06, 0, 0.06])
        # ax.set_yticks([-0.06, 0, 0.06])
        # ax.set_zticks([-0.06, 0, 0.06])
        # ax.scatter([x[0] for x in new_vertices_positions], [x[1_Energy_Contrast] for x in new_vertices_positions],
        #            [x[2] for x in new_vertices_positions], c='blue', marker='o', linestyle='-',
        #            alpha=0.6, s=12)
        # plt.savefig('/home/workuser/Desktop/TestSoftware/Voronoi/cc3_removed.jpg', dpi=800)

        Hierarchy = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self._threshold,
            metric='euclidean',
            linkage='ward'
        )
        cluster = Hierarchy.fit(new_vertices_positions)
        labels = cluster.labels_

        if verbose:
            print(f"Number of clusters: {len(np.unique(labels))}")
            print(f"Labels : {labels}\n")

        # Link the cluster.labels_ to new_vertices_positions
        # and return the vertex which is the farthest to the host atoms.
        cluster_vertices = []
        cluster_vertices_distance = []
        for i in range(len(np.unique(labels))):
            index = np.where(labels == i)
            cluster_vertices.append(
                new_vertices_positions[
                    index[0][
                        np.argmax(
                            np.array(new_vertices_distance)[index]
                        )
                    ]
                ]
            )
            cluster_vertices_distance.append(
                new_vertices_distance[
                    index[0][
                        np.argmax(
                            np.array(new_vertices_distance)[index]
                        )
                    ]
                ]
            )

        # Use Hierarchy cluster twice to decrease the number of vertices.
        if len(cluster_vertices) > 2:
            cluster_vertices2 = []
            cluster2 = Hierarchy.fit(cluster_vertices)
            labels2 = cluster2.labels_
            for i in range(len(np.unique(labels2))):
                index2 = np.where(labels2 == i)
                cluster_vertices2.append(
                    cluster_vertices[
                        index2[0][
                            np.argmax(
                                np.array(cluster_vertices_distance)[index2]
                            )
                        ]
                    ]
                )
            if len(cluster_vertices2) >= final_vertices_threshold:
                centroid_dis_list = [round(distance_between_dot(self._centroid.reshape((1, 3)), j.reshape((1, 3))), 4)
                                     for j
                                     in cluster_vertices]
                sorted_list = sorted(enumerate(centroid_dis_list), key=lambda x: x[1])
                cluster_vertices2 = [cluster_vertices2[i[0]] for i in sorted_list[:final_vertices_threshold]]
            return np.array(cluster_vertices2)

        # To determine whether the number of vertices is too large.
        # ==============================================================
        if len(cluster_vertices) >= final_vertices_threshold:
            centroid_dis_list = [round(distance_between_dot(self._centroid.reshape((1, 3)), j.reshape((1, 3))), 4) for j
                                 in cluster_vertices]
            sorted_list = sorted(enumerate(centroid_dis_list), key=lambda x: x[1])
            cluster_vertices = [cluster_vertices[i[0]] for i in sorted_list[:final_vertices_threshold]]
        return np.array(cluster_vertices)
