import MDAnalysis as mda
import os
import platform

from MDAnalysis.analysis import rms

CURRENT_OS = platform.system()


class Trajectory:
    """
    用于处理轨迹文件的类
    """

    def __init__(self, topo, traj, scratch_dir):
        if topo is None:
            self._u = mda.Universe(traj)
        else:
            self._u = mda.Universe(topo, traj)
        self._scratch_dir = scratch_dir
        self._frame_num = len(self._u.trajectory)

    def _frame_to_indices(self, num_confs=200, rmsd_threshold=None, frame_range=None):
        """
        Extract frames from trajectory file and convert them to mol files.

        Parameters
        --------------------
        num_confs: int
            The number of conformers to be extracted.

        rmsd_threshold: float
            The threshold of rmsd among each frame.

        frame_range: list, optional
            The range of frames to be extracted.

        """
        print(f'-------------Total Frames: {self._frame_num}----------------')
        if frame_range is None:
            frame_range = [0, self._frame_num]
        if rmsd_threshold is None:
            step = self._frame_num // num_confs
            frame_indices = [index for index in range(frame_range[0], frame_range[1], step)]
            return frame_indices
        else:
            seg = self._u.select_atoms(f'segid {self._u.segments.segids[0]}')
            rmsd = rms.RMSD(seg, seg).run()
            bools = rmsd.results.rmsd.T[-1] < rmsd_threshold
            fiter = self._u.trajectory[bools]
            frame_indices = [ts.frame for ts in fiter]

            # Frame_indices should be a subset of frame_range
            frame_indices = [index for index in frame_indices if frame_range[0] <= index < frame_range[1]]

            # Length indices should be less than num_confs
            if len(frame_indices) > num_confs:
                frame_indices = frame_indices[:num_confs]
            return frame_indices

    def frame_to_mol(self, num_confs, rmsd_threshold, frame_range=None):
        """
        Extract frames from trajectory file and convert them to mol files.

        Parameters
        --------------------
        num_confs: int
            The number of conformers to be extracted.

        rmsd_threshold: float
            The threshold of rmsd among each frame.

        frame_range: list, optional
            The range of frames to be extracted.

        Returns
        --------------------
        None:
            A series of .mol files to 'scratch_dir'

        """
        frame_indices = self._frame_to_indices(num_confs, rmsd_threshold, frame_range)
        if not os.path.exists(self._scratch_dir):
            os.makedirs(self._scratch_dir)
        for i, index in enumerate(frame_indices):
            self._u.trajectory[index]
            seg = self._u.select_atoms(f'segid {self._u.segments.segids[0]}')
            seg.write(f'{self._scratch_dir}/{i}.xyz')

            if CURRENT_OS == 'Linux':
                os.system(f'obabel {self._scratch_dir}/{i}.xyz -O {self._scratch_dir}/{i}.mol  > /dev/null 2>&1')
                os.system(f'rm {self._scratch_dir}/{i}.xyz')
            elif CURRENT_OS == 'Windows':
                os.system(f'obabel {self._scratch_dir}/{i}.xyz -O {self._scratch_dir}/{i}.mol')
                os.system(f'rm {self._scratch_dir}/{i}.xyz')
        return None

    def __len__(self):
        return self._frame_num

    def __repr__(self):
        return f'Trajectory({self._u})'
