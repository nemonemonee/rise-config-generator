import torch as t
import numpy as np
from rise import Rise, RiseFrame
from sim.builder import SimBuilder
from typing import Tuple, List, Any


def get_callback(
    material_num,
    robot_index,
    controller,
    records,
    center_of_masses,
    fitness_critique_observe_frames,
    fitness_critique_observe_interval,
):
    if controller is not None:
        raise ValueError("Currently controllers are not supported.")

    record_index = 0
    last_record_time = 0

    def callback(
        time: float, frame: RiseFrame, expansion_signals: Any, rotation_signals: Any
    ):
        nonlocal record_index, last_record_time
        if record_index < fitness_critique_observe_frames and (
            last_record_time == 0
            or time - last_record_time >= fitness_critique_observe_interval
        ):
            voxels = frame.voxels()
            positions = np.array([v[7].tolist() for v in voxels])
            material = np.array([v[4] for v in voxels])
            record = np.zeros([voxels.shape[0], material_num], dtype=float)
            record[list(range(voxels.shape[0])), material - 1] = 1
            records[robot_index][record_index] = t.from_numpy(
                np.concatenate([positions, record], axis=1)
            )
            center_of_masses[robot_index][record_index] = t.mean(
                t.from_numpy(positions), dim=0
            )
            last_record_time = time
            record_index += 1

    return callback


class RiseEnv:
    def __init__(
        self,
        devices: List[int],
        env_config: str,
        voxel_size: float,
        material_num: int,
        material_is_rigid: List[bool],
        voxel_grid_size: Tuple[int, int, int],
        result_path: str = "",
        record_path: str = "",
    ):
        self.rise = Rise(devices=devices)
        self.env_config = env_config
        self.voxel_size = voxel_size
        self.material_num = material_num
        self.material_is_rigid = material_is_rigid
        self.voxel_grid_size = voxel_grid_size
        self.result_path = result_path
        self.record_path = record_path

    def run_sims(
        self,
        generation: int,
        robots: t.Tensor,
        controller: Any = None,
        builder_kwargs: dict = None,
        fitness_critique_observe_frames: int = 10,
        fitness_critique_observe_interval: float = 0.1,
        save_record: bool = False,
        record_buffer_size: int = 100,
    ):
        """
        Args:
            generation: A unique generation number, robot results will be saved as
                "{generation}_{index}.result" under result_path, and records will
                be saved as "{generation}_{index}.h5_history" under record_path
            robots: Robot configuration tensor of shape [N, x_size * y_size * z_size, conf_dim]
            controller: A controller model which takes in observations of shape
                [N*, voxels, observe_dim] and returns actions of shape [N*, voxels, 2],
                for the last dimension, first is expansion signal and second is rotation
                signal.
        Returns:
            A four element tuple,
                first: A list of bools, length N, indicating whether simulation is
                    successful, unsuccessful experiments will not have a result file,
                    the record file may exist and contains information up until the error
                    in simulation occurs.
                second: A list of list of tensors, length N,
                    inner list has length fitness_critique_observe_frames,
                    each element in the list may be None, or a tensor of shape [voxels, observe_dim],
                third: A list of list of numpy arrays, length N,
                    inner list has length fitness_critique_observe_frames,
                    each element in the list may be None, or a tensor of shape [3],
                fourth: A list of robot configuration rsc file content, length N
        Note:
            N: robot num (batch size)

            N*: Any value from 1 to N

            x_size, y_size, z_size: Defined by voxel_grid_size

            voxels: Number of robot voxels

            conf_dim: See sim.builder.SimBuilder

            observe_dim: Equal to material_num + 3, first 3 dim are positions XYZ

        Note:
            Robots tensor is row ordered, contiguous in x, X axis with minimum y
            and z comes first. See sim.builder.SimBuilder

        TODO: add terrain observation to observe_dim
        """
        records = [
            [None] * fitness_critique_observe_frames for _ in range(robots.shape[0])
        ]
        center_of_masses = [
            [None] * fitness_critique_observe_frames for _ in range(robots.shape[0])
        ]
        configs = []
        robot_configs = []
        callbacks = []

        builder_kwargs = builder_kwargs or {}
        robots = robots.detach().cpu()
        for idx, robot in enumerate(robots):
            builder = SimBuilder(self.voxel_size)
            robot_config = builder.build(
                self.material_num,
                self.material_is_rigid,
                self.voxel_grid_size,
                robot,
                sim_name=f"{generation}_{idx}",
                result_path=self.result_path,
                record_path=self.record_path,
                save_history=save_record,
                save_h5_history=save_record,
                **builder_kwargs,
            )
            with open("robot.rsc", "w") as file:
                file.write(robot_config)
            configs.append([self.env_config, robot_config])
            robot_configs.append(robot_config)
            callbacks.append(
                get_callback(
                    self.material_num,
                    idx,
                    controller,
                    records,
                    center_of_masses,
                    fitness_critique_observe_frames,
                    fitness_critique_observe_interval,
                )
            )
        result = self.rise.run_sims(
            configs,
            callbacks,
            record_buffer_size=1 if not save_record else record_buffer_size,
            save_result=True,
            save_record=save_record,
        )
        return result, records, center_of_masses, robot_configs
