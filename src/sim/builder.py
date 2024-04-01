import collections
import os
import cc3d
import numpy as np
import torch
import lxml.etree as etree
from typing import Tuple, List
from numpy import linspace
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


class SimBuilder:
    def __init__(self, voxel_size: float = 0.01):
        self.voxel_size = voxel_size

    def build(
        self,
        material_num: int,
        material_is_rigid: List[bool],
        voxel_grid_size: Tuple[int, int, int],
        voxels: torch.Tensor,
        threshold: float = 0.5,
        min_size: int = 10,
        search_radius: int = 5,
        segment_resolution: int = 1024,
        sim_name: str = "robot",
        result_path: str = "",
        record_path: str = "",
        save_history: bool = False,
        save_h5_history: bool = True,
    ):
        """
        Args:
            material_num: Number of materials.
            material_is_rigid: List of bools indicating whether the material is rigid
            voxel_grid_size: Size of the voxel grid (x_size, y_size, z_size)
            voxels: A tensor of shape [x_size * y_size * z_size, material_num + 7],
                First dimension is XYZ flattened.
                Second dimension is ordered as follows:
                    0: null material, weight in range [0, 1]
                    1 -> material_num: materials, weight in range [0, 1]
                    material_num + 1: segment ids, normalized to range [0, 1]
                    material_num + 2: hinge joint presence, in range [0, 1]
                    material_num + 3: ball and socket joint presence, in range [0, 1]
                    material_num + 4 -> material_num + 6: unit hinge axis


        Returns:
            A rsc config string for simulation
        """

        ROBOT_RSC = """
        <RSC>
            <Structure>
                <Bodies>
                    {}
                </Bodies>
                <Constraints>
                    {}
                </Constraints>
            </Structure>
            <Simulator>
                <Signal>
                    <ExpansionNum>0</ExpansionNum>
                    <RotationNum>{}</RotationNum>
                </Signal>
                <RecordHistory>
                    <RecordStepSize>200</RecordStepSize>
                    <RecordVoxel>1</RecordVoxel>
                    <RecordLink>0</RecordLink>
                    <RecordFixedVoxels>0</RecordFixedVoxels>
                </RecordHistory>
            </Simulator>
            <Save>
                {}
            </Save>
        </RSC>
        """
        layer_size = voxel_grid_size[0] * voxel_grid_size[1]
        total_size = layer_size * voxel_grid_size[2]
        if voxels.shape[0] != total_size or voxels.shape[1] != material_num + 7:
            raise ValueError("voxels size doesn't match")
        voxels = voxels.cpu().numpy()
        # convert first dimension from XYZ order to ZYX order
        voxels = (
            voxels.reshape(
                voxel_grid_size[2],
                voxel_grid_size[1],
                voxel_grid_size[0],
                voxels.shape[-1],
            )
            .transpose(2, 1, 0, 3)
            .reshape(-1, voxels.shape[-1])
        )
        material_id, segment_id, segment_is_rigid, body_config = self.get_body_config(
            material_num=material_num,
            material_is_rigid=material_is_rigid,
            voxel_grid_size=voxel_grid_size,
            voxels=voxels,
            segment_resolution=segment_resolution,
        )
        constraint_configs = self.get_constraint_configs(
            material_num=material_num,
            voxel_grid_size=voxel_grid_size,
            voxels=voxels,
            segment_id=segment_id,
            segment_is_rigid=segment_is_rigid,
            threshold=threshold,
            min_size=min_size,
            search_radius=search_radius,
        )
        save_config = self.get_save_config(
            sim_name=sim_name,
            result_path=result_path,
            record_path=record_path,
            save_history=save_history,
            save_h5_history=save_h5_history,
        )
        robot_rsc = ROBOT_RSC.format(
            body_config,
            "\n".join(constraint_configs),
            len(constraint_configs),
            save_config,
        )
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.fromstring(robot_rsc, parser=parser)
        return etree.tostring(tree, pretty_print=True, encoding=str)

    def get_body_config(
        self,
        material_num: int,
        material_is_rigid: List[bool],
        voxel_grid_size: Tuple[int, int, int],
        voxels: np.ndarray,
        segment_resolution: int = 1024,
    ):
        BODY_CONFIG = """
        <Body ID="1">
            <Orientation>0,0,0,1</Orientation>
            <OriginPosition>0,0,0</OriginPosition>
            {}
            <MaterialID>
                {}
            </MaterialID>
            <SegmentID>
                {}
            </SegmentID>
            <SegmentType>
                {}
            </SegmentType>
        </Body>
        """
        SHAPE_TEMPLATE = """
            <X_Voxels>{}</X_Voxels>
            <Y_Voxels>{}</Y_Voxels>
            <Z_Voxels>{}</Z_Voxels>
        """
        LAYER_TEMPLATE = "<Layer>{}</Layer>"

        layer_size = voxel_grid_size[0] * voxel_grid_size[1]
        shape = SHAPE_TEMPLATE.format(*voxel_grid_size)

        material_indices = np.argmax(voxels[:, : material_num + 1], axis=1)
        # perform segmentation
        segment = (
            voxels[:, material_num + 1].clip(0, 1).reshape(-1, 1) * segment_resolution
        )
        kde = KernelDensity(kernel="gaussian", bandwidth=3).fit(segment)
        s = linspace(0, segment_resolution, segment_resolution + 1)
        estimate = kde.score_samples(s.reshape(-1, 1))

        # for showing intervals
        # plt.plot(s, estimate)
        # plt.show()

        minima = argrelextrema(estimate, np.less)[0]

        # create bins using minima
        bin_num = minima.shape[0] + 1
        segment_bin_indices = np.zeros(voxels.shape[0], dtype=int)
        segment = segment.reshape(-1)
        if len(minima) > 0:
            for i in range(bin_num):
                if i == 0:
                    segment_bin_indices[segment < minima[0]] = i
                elif i == bin_num - 1:
                    segment_bin_indices[segment > minima[i - 1]] = i
                else:
                    segment_bin_indices[
                        np.logical_and(segment > minima[i - 1], segment < minima[i])
                    ] = i

        # check segment consistency
        # only keep the segment if > 50% material_ids are the same,
        # otherwise set segment to 0 and material to 0
        segment_material_indices = {}
        for material_index, segment_bin_index in zip(
            material_indices, segment_bin_indices
        ):
            if segment_bin_index not in segment_material_indices:
                segment_material_indices[segment_bin_index] = []
            segment_material_indices[segment_bin_index].append(material_index)

        invalid_segment_indices = set()
        segment_to_material = {}
        segment_is_rigid = {}
        material_is_rigid = [False] + material_is_rigid
        for (
            segment_id,
            per_segment_material_indices,
        ) in segment_material_indices.items():
            counter = collections.Counter(per_segment_material_indices)
            most_common_material, most_common_count = counter.most_common(1)[0]
            if most_common_count / len(per_segment_material_indices) < 0.5:
                invalid_segment_indices.add(segment_id)
            else:
                segment_to_material[segment_id] = most_common_material
                segment_is_rigid[segment_id] = material_is_rigid[most_common_material]

        # zero out materials in segment 0
        mask = segment_bin_indices == 0
        material_indices[mask] = 0
        segment_bin_indices[mask] = 0

        for invalid_segment_index in invalid_segment_indices:
            mask = segment_bin_indices == invalid_segment_index
            material_indices[mask] = 0
            segment_bin_indices[mask] = 0

        for segment_index, material_index in segment_to_material.items():
            mask = segment_bin_indices == segment_index
            material_indices[mask] = material_index

        material_id, segment_id, segment_type = [], [], []
        material_is_rigid = np.array([False] + material_is_rigid, dtype=int)
        for z in range(voxel_grid_size[2]):
            offset = z * layer_size
            material_id.append(
                LAYER_TEMPLATE.format(
                    ",".join(map(str, material_indices[offset : offset + layer_size]))
                )
            )
            segment_id.append(
                LAYER_TEMPLATE.format(
                    ",".join(
                        map(str, segment_bin_indices[offset : offset + layer_size])
                    )
                )
            )
            segment_type.append(
                LAYER_TEMPLATE.format(
                    ",".join(
                        map(
                            str,
                            material_is_rigid[
                                material_indices[offset : offset + layer_size]
                            ],
                        )
                    )
                )
            )
        return (
            material_indices,
            segment_bin_indices,
            segment_is_rigid,
            BODY_CONFIG.format(
                shape,
                "\n".join(material_id),
                "\n".join(segment_id),
                "\n".join(segment_type),
            ),
        )

    def get_constraint_configs(
        self,
        material_num: int,
        voxel_grid_size: Tuple[int, int, int],
        voxels: np.ndarray,
        segment_id: np.ndarray,
        segment_is_rigid: dict,
        threshold: float = 0.5,
        min_size: int = 10,
        search_radius: int = 5,
    ):
        HINGE_CONSTRAINT_TEMPLATE = """
        <Constraint>
            <Type>HINGE_JOINT</Type>
            <RigidBodyA>
                <BodyID>1</BodyID>
                <SegmentID>{}</SegmentID>
                <Anchor>{},{},{}</Anchor>
            </RigidBodyA>
            <RigidBodyB>
                <BodyID>1</BodyID>
                <SegmentID>{}</SegmentID>
                <Anchor>{},{},{}</Anchor>
            </RigidBodyB>
            <HingeRotationSignalID>{}</HingeRotationSignalID>
            <HingeAAxis>{}, {}, {}</HingeAAxis>
            <HingeBAxis>{}, {}, {}</HingeBAxis>
        </Constraint>
        """
        BALL_AND_SOCKET_CONSTRAINT_TEMPLATE = """
        <Constraint>
            <Type>BALL_AND_SOCKET_JOINT</Type>
            <RigidBodyA>
                <BodyID>1</BodyID>
                <SegmentID>{}</SegmentID>
                <Anchor>{},{},{}</Anchor>
            </RigidBodyA>
            <RigidBodyB>
                <BodyID>1</BodyID>
                <SegmentID>{}</SegmentID>
                <Anchor>{},{},{}</Anchor>
            </RigidBodyB>
        </Constraint>
        """
        hinge_joint_presence = voxels[:, material_num + 2] > threshold
        ball_and_socket_joint_presence = voxels[:, material_num + 3] > threshold
        z_coords, y_coords, x_coords = np.meshgrid(
            np.arange(voxel_grid_size[2]),
            np.arange(voxel_grid_size[1]),
            np.arange(voxel_grid_size[0]),
            indexing="ij",
        )
        reshape = (voxel_grid_size[2], voxel_grid_size[1], voxel_grid_size[0])
        segment_id = segment_id.reshape(reshape)
        hinge_labels, hinge_label_num = cc3d.connected_components(
            hinge_joint_presence.reshape(reshape),
            connectivity=26,
            return_N=True,
            out_dtype=np.uint32,
        )
        hinge_label_count = np.bincount(
            hinge_labels.reshape(-1), minlength=hinge_label_num
        )
        # Ignore label 0, which is non-occupied space
        hinge_joint_positions = []
        for label in range(1, hinge_label_num):
            if hinge_label_count[label] > min_size:
                mask = hinge_labels == label
                hinge_joint_positions.append(
                    (
                        np.mean(x_coords[mask]),
                        np.mean(y_coords[mask]),
                        np.mean(z_coords[mask]),
                    )
                )

        ball_and_socket_labels, ball_and_socket_num = cc3d.connected_components(
            ball_and_socket_joint_presence.reshape(reshape),
            connectivity=26,
            return_N=True,
            out_dtype=np.uint32,
        )
        ball_and_socket_label_count = np.bincount(
            ball_and_socket_labels.reshape(-1), minlength=ball_and_socket_num
        )
        ball_and_socket_positions = []
        for label in range(1, ball_and_socket_num):
            if ball_and_socket_label_count[label] > min_size:
                mask = ball_and_socket_labels == label
                ball_and_socket_positions.append(
                    (
                        np.mean(x_coords[mask]),
                        np.mean(y_coords[mask]),
                        np.mean(z_coords[mask]),
                    )
                )

        # Add constraints by searching for nearby segments
        constraints = []
        constrained_segment_pairs = set()
        hinge_rotation_id = 0
        for hinge_joint_position in hinge_joint_positions:
            x, y, z = hinge_joint_position
            distance = np.linalg.norm(
                np.stack((x_coords - x, y_coords - y, z_coords - z), axis=0), axis=0
            )
            in_range_mask = distance < search_radius
            in_range_distance = distance[in_range_mask]
            in_range_segment_id = segment_id[in_range_mask]
            segment_distances = {}
            for seg_dist, seg_id in zip(in_range_distance, in_range_segment_id):
                if (
                    seg_id > 0
                    and seg_id in segment_is_rigid
                    and segment_is_rigid[seg_id]
                ):
                    if seg_id not in segment_distances:
                        segment_distances[seg_id] = []
                    segment_distances[seg_id].append(seg_dist)
            for seg_id, seg_dists in segment_distances.items():
                segment_distances[seg_id] = np.min(seg_dists)

            if len(segment_distances) >= 2:
                sorted_distances = sorted(segment_distances.items(), key=lambda x: x[1])
                offset = (
                    int(x)
                    + int(y) * voxel_grid_size[0]
                    + int(z) * (voxel_grid_size[0] * voxel_grid_size[1])
                )
                hinge_axis = voxels[offset, material_num + 4 : material_num + 7]
                hinge_axis_norm = np.linalg.norm(hinge_axis)
                if hinge_axis_norm > 0.01:
                    hinge_axis = hinge_axis / hinge_axis_norm
                    # Get the 2 closest segments
                    segments = (sorted_distances[0][0], sorted_distances[1][0])
                    if segments not in constrained_segment_pairs:
                        constrained_segment_pairs.add(segments)
                        constrained_segment_pairs.add((segments[1], segments[0]))
                        constraints.append(
                            HINGE_CONSTRAINT_TEMPLATE.format(
                                segments[0],
                                hinge_joint_position[0] * self.voxel_size,
                                hinge_joint_position[1] * self.voxel_size,
                                hinge_joint_position[2] * self.voxel_size,
                                segments[1],
                                hinge_joint_position[0] * self.voxel_size,
                                hinge_joint_position[1] * self.voxel_size,
                                hinge_joint_position[2] * self.voxel_size,
                                hinge_rotation_id,
                                hinge_axis[0],
                                hinge_axis[1],
                                hinge_axis[2],
                                -hinge_axis[0],
                                -hinge_axis[1],
                                -hinge_axis[2],
                            )
                        )
                        hinge_rotation_id += 1

        for ball_and_socket_joint_position in ball_and_socket_positions:
            x, y, z = ball_and_socket_joint_position
            distance = np.linalg.norm(
                np.stack((x_coords - x, y_coords - y, z_coords - z), axis=0), axis=0
            )
            in_range_mask = distance < search_radius
            in_range_distance = distance[in_range_mask]
            in_range_segment_id = segment_id[in_range_mask]
            segment_distances = {}
            for seg_dist, seg_id in zip(in_range_distance, in_range_segment_id):
                if (
                    seg_id > 0
                    and seg_id in segment_is_rigid
                    and segment_is_rigid[seg_id]
                ):
                    if seg_id not in segment_distances:
                        segment_distances[seg_id] = []
                    segment_distances[seg_id].append(seg_dist)
            for seg_id, seg_dists in segment_distances.items():
                segment_distances[seg_id] = np.min(seg_dists)

            if len(segment_distances) >= 2:
                sorted_distances = sorted(segment_distances.items(), key=lambda x: x[1])
                # Get the 2 closest segments
                segments = (sorted_distances[0][0], sorted_distances[1][0])
                if segments not in constrained_segment_pairs:
                    constrained_segment_pairs.add(segments)
                    constrained_segment_pairs.add((segments[1], segments[0]))
                    constraints.append(
                        BALL_AND_SOCKET_CONSTRAINT_TEMPLATE.format(
                            segments[0],
                            ball_and_socket_joint_position[0] * self.voxel_size,
                            ball_and_socket_joint_position[1] * self.voxel_size,
                            ball_and_socket_joint_position[2] * self.voxel_size,
                            segments[1],
                            ball_and_socket_joint_position[0] * self.voxel_size,
                            ball_and_socket_joint_position[1] * self.voxel_size,
                            ball_and_socket_joint_position[2] * self.voxel_size,
                        )
                    )

        return constraints

    def get_save_config(
        self,
        sim_name: str = "robot",
        result_path: str = "",
        record_path: str = "",
        save_history: bool = False,
        save_h5_history: bool = True,
    ):
        SAVE_CONFIG = """
        <ResultPath>{}</ResultPath>
        <Record>
            {}
        </Record>
        """
        TXT_RECORD_CONFIG = """
        <Text>
            <Rescale>0.001</Rescale>
            <Path>{}</Path>
        </Text>
        """
        H5_RECORD_CONFIG = """
        <HDF5>
            <Path>{}</Path>
        </HDF5>
        """
        records = []
        if save_history:
            records.append(
                TXT_RECORD_CONFIG.format(
                    os.path.join(record_path, f"{sim_name}.history")
                )
            )
        if save_h5_history:
            records.append(
                H5_RECORD_CONFIG.format(
                    os.path.join(record_path, f"{sim_name}.h5_history")
                )
            )
        return SAVE_CONFIG.format(
            os.path.join(result_path, f"{sim_name}.result"), "\n".join(records)
        )
