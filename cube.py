# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 下午4:32
# @Author  : wzd
# @File    : cube.py
from scipy.spatial import distance


class Grid:
    """
    class 'Grid'.

    Describe a 'Grid' instance.

    """

    def __init__(self, length, center_position, is_outer=False):
        """
        Initialize a class `Grid` instance.

        Parameter
        ------


        """
        self._length = length
        self._center_position = center_position
        self._is_outer = is_outer

    def get_cube_size(self):
        if self._length <= 0:
            raise ValueError(f'This grid is invalid with length {self._length}.')
        else:
            return self._length

    def get_cube_position(self):
        return self._center_position

    def get_distance_from_dot(self, dot_position):
        return distance.euclidean(self.get_cube_position(), dot_position)


class Cube:
    """
    class 'Cube'.

    Describe a 'Cube' instance.

    """

    def __init__(self, grid_list):
        """
        Initialize a class `Cube` instance.

        Parameter
        ------
        """
        pass
