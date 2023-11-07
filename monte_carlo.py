# -*- coding: utf-8 -*-
# @Time    : 6/14/22 9:21 PM
# @Author  : name
# @File    : monte_carlo.py
"""
-----------------------------------------------------------------------------------------
ATTENTION: Old version has been moved to 'scratch/copy_main/monte_carlo.py' file.
           New version of monte_carlo has moved functions for testing.
-----------------------------------------------------------------------------------------
move guest in random method.

function:
    1_Energy_Contrast. MCMC move
    2. REMC move
    3. binding pocket calculation
    4. different initial conformers sampling
"""
import numpy as np
import random

from typing import Optional
from .forcefield import potential
from .forcefield.potential import SimpleLJPotential, AmberPotential
from .molecule import Molecule, HostMolecule, GuestMolecule, HostGuestComplex
from .utilities import distance_between_dot, exam_move, norm_vector


boltzmann_constant = 1.380649e-23


def define_box(molecule: Molecule, cutoff: float = 5.0) -> np.ndarray:
    """
    Apply a box around the complex to restrict the movement of the guest.

    Parameters
    ------
    molecule: 'HostMolecule' or 'HostGuestComplex'

    cutoff: float
        The cutoff distance of the box, default to be 5 angstrom.

    """
    box_length = molecule.cal_max_diameter() + cutoff
    return box_length


class MonteCarlo:
    """
    Move the guest molecule in Monte Carlo way and return the lowest
    energy conformer.

    """

    def __init__(
            self,
            host: HostMolecule,
            guest: Optional[list, tuple] = None,
            cutoff: int = -2,
            temperature: int = 300,
            step_size: float = 0.5,
            rotation_step_size: float = 0.2,
            self_rotation_step_size: float = 0.2,
            max_moves: int = 3000,
            max_conformers: int = 300,
            verbose: bool = True,
            calculator: 'potential' = SimpleLJPotential,
            random_seed: int = 1000,
            mc_constant: float = 2.0
    ):
        """
        Parameters
        ----------
        host: 'HostMolecule'
            The host molecule.

        guest: 'GuestMolecule'
            The guest molecule.

        cutoff: float
            The cutoff distance of the box, default to be 5 angstrom.

        temperature: float
            The temperature of the system, default to be 300 K.

        step_size: float
            The step size of the translation, default to be 0.1_Energy_Contrast angstrom.

        rotation_step_size: float
            The step size of the rotation, default to be 0.1_Energy_Contrast degree.

        self_rotation_step_size: float
            The step size of the self-rotation, default to be 0.1_Energy_Contrast degree.

        max_moves: int
            The maximum number of moves, default to be 10000.

        max_conformers: int
            The maximum number of conformers, default to be 300.

        verbose: bool
            Whether to print the progress of the optimization, default to be False.

        calculator: class
            The calculator to calculate the potential, default to be SimpleLJPotential.

        random_seed: int
            The random seed of the random number generator, default to be 1000.

        mc_constant: float
            The constant of the metropolis MC, default to be 2.0.

        """
        self._boltzmann_constant = boltzmann_constant
        self._host = host
        self._box_length = define_box(host, cutoff)
        self._cutoff = cutoff
        self._temperature = temperature
        self._step_size = step_size
        self._rotation_step_size = rotation_step_size
        self._self_rotation_step_size = self_rotation_step_size
        self._max_moves = max_moves
        self._verbose = verbose
        self._calculator = calculator
        self._boltzmann_constant = boltzmann_constant
        self._max_conformers = max_conformers
        self._mc_constant = mc_constant
        if isinstance(guest, list or tuple):
            self._guest = guest
        else:
            raise TypeError("The guest should be loaded as a list or tuple.")
        np.random.seed(random_seed)
        random.seed(random_seed)

    def _exam_move(self, energy_old: float, energy_new: float) -> bool:
        if energy_new < energy_old:
            return True
        else:
            value = np.exp(-self._mc_constant * (energy_new - energy_old))
            # TEST
            # value = np.exp(-(energy_new - energy_old) / 4)
            random_value = random.random()
            # TEST
            # print(f"value : {value}; random_value: {random_value}\n")
            if value > random_value:
                return True
            else:
                return False

    def _exam_box_boundary(self, molecule: Molecule) -> bool:
        """
        Exam whether the molecule is in the box.

        Parameters
        ----------
        molecule: 'Molecule'
            The molecule to be exam.

        """
        box_limit = self._box_length / 2
        x_min, y_min, z_min = -box_limit, -box_limit, -box_limit
        x_max, y_max, z_max = box_limit, box_limit, box_limit

        positions = molecule.get_positions()
        for i in range(len(positions)):
            x, y, z = positions[i]
            if x < x_min or x > x_max:
                return False
            elif y < y_min or y > y_max:
                return False
            elif z < z_min or z > z_max:
                return False
        return True

    def _cal_potential(self, host, guest) -> float:
        """
        Calculate the simplest LJ potential of the guest molecule.

        Returns
        ------
        'float'
            The potential energy of the complex.

        """
        lj_potential = self._calculator(host, guest)
        return lj_potential.cal_potential()

    @staticmethod
    def _norm_vector(vector):
        return np.array(vector) / np.linalg.norm(vector)

    def move_guest(
            self,
            guest_positions: np.ndarray = None
    ):
        """
        Move the guest molecule in Monte Carlo way and return the lowest
        energy conformers.

        Yields
        ------
        'HostGuestComplex'

        """
        # Move host centroid and guest centroid to the same position.
        host = self._host.translate_to_new_origin()
        if len(self._guest) == 0:
            raise ValueError("The guest list is empty.")

        elif len(self._guest) == 1:
            if guest_positions:
                guest = self._guest[0].translate_to_new_centroid(guest_positions[0])
            else:
                # Move host centroid and guest centroid to the same position.
                guest = self._guest[0].translate_to_new_centroid([0, 0, 0])

            potentials = [self._cal_potential(host, guest)]
            test_potential = [self._cal_potential(host, guest)]

            times = 1
            conformers = 0
            while conformers < self._max_conformers:
                if times >= self._max_moves:
                    raise RuntimeError("The maximum number of moves is reached.")
                # Get the random vector.
                rand_vector = self._norm_vector(np.random.rand(3) * 2 - 1)

                # Get the translation vector.
                translation_vector = rand_vector * self._step_size
                new_guest = guest.move_guest_molecule(translation_vector)

                # Get the rotation angle in [-1_Energy_Contrast, 1_Energy_Contrast].
                rand_angle = random.random() * 2 - 1
                rotation_angle = rand_angle * self._rotation_step_size
                # Get the rotation axis.
                # Begin with the com of guest molecule and end with the random dot on the unit sphere.
                theta = 2 * np.pi * random.random()
                phi = np.arccos(2 * random.random() - 1)
                rotation_axis = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
                new_guest = new_guest.rotate_guest_molecule(rotation_axis, rotation_angle)

                # Get the new potential.
                new_potential = self._cal_potential(host, new_guest)
                potentials.append(new_potential)

                # Check the move.
                if self._exam_move(potentials[-2], potentials[-1]):
                    conformers += 1
                    # If the distance between guest centroid and host centroid is larger than
                    # the pore radii of host molecule, it means the guest molecule is out of the host molecule.
                    # So we pass it.
                    test_potential.append(new_potential)
                    yield HostGuestComplex.init_from_molecule(host, new_guest)
                else:
                    pass
                times += 1

            if self._verbose:
                print(f"LJ-Potential : {potentials}")
                print(f"Passed potential : {test_potential}")

        else:
            if guest_positions:
                guests = [g.translate_to_new_centroid(guest_positions[i]) for i, g in enumerate(self._guest)]
            else:
                guests = [g.translate_to_new_centroid([0, 0, 0]) for g in self._guest]
            potentials = [self._cal_potential(host, g) for g in guests]
            test_potential = [self._cal_potential(host, g) for g in guests]

            times = 1
            conformers = 0
            while conformers < self._max_conformers:
                if times >= self._max_moves:
                    raise RuntimeError("The maximum number of moves is reached.")
                # Get the random vector.
                rand_vector = self._norm_vector(np.random.rand(3) * 2 - 1)

                # Get the translation vector.
                translation_vector = rand_vector * self._step_size
                new_guests = [g.move_guest_molecule(translation_vector) for g in guests]

                # Get the rotation angle in [-1_Energy_Contrast, 1_Energy_Contrast].
                rand_angle = random.random() * 2 - 1
                rotation_angle = rand_angle * self._rotation_step_size
                # Get the rotation axis.
                # Begin with the com of guest molecule and end with the random dot on the unit sphere.
                theta = 2 * np.pi * random.random()
                phi = np.arccos(2 * random.random() - 1)
                rotation_axis = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
                new_guests = [g.rotate_guest_molecule(rotation_axis, rotation_angle) for g in new_guests]

                # Get the new potential.
                new_potentials = [self._cal_potential(host, g) for g in new_guests]
                potentials.extend(new_potentials)

                # Check the move.
                for i, new_potential in enumerate(new_potentials):
                    if self._exam_move(potentials[-len(new_potentials) + i], new_potential):
                        conformers += 1
                        # If the distance between guest centroid and host centroid is larger than
                        # the pore radii of host molecule, it means the guest molecule is out of the host molecule.
                        # So we pass it.
                        test_potential.append(new_potential)
                        yield
            pass

    def move_guest_print_all(self):
        # Move host centroid and guest centroid to the same position.
        host = self._host.translate_to_new_origin()
        guest = self._guest.translate_to_new_centroid([0, 0, 0])

        times = 0
        while times < self._max_moves:
            # Get the random vector.
            rand_vector = self._norm_vector(np.random.rand(3) * 2 - 1)

            # Get the translation vector.
            translation_vector = rand_vector * self._step_size
            new_guest = guest.move_guest_molecule(translation_vector)

            # Get the rotation angle in [-1_Energy_Contrast, 1_Energy_Contrast].
            rand_angle = random.random() * 2 - 1
            rotation_angle = rand_angle * self._rotation_step_size

            # Get the rotation axis.
            # Begin with the com of guest molecule and end with the random dot on the unit sphere.
            theta = 2 * np.pi * random.random()
            phi = np.arccos(2 * random.random() - 1)
            rotation_axis = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
            new_guest = new_guest.rotate_guest_molecule(rotation_axis, rotation_angle)

            yield HostGuestComplex.init_from_molecule(host, new_guest)

            times += 1
