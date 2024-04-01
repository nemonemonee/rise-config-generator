import argparse
import numpy as np
from src.templates import *
from src.load_robot import Robot


def generate_body(voxels: object, segments: object, rigid_material: object = 2) -> object:
    input_shape = voxels.shape
    layer_size = input_shape[0] * input_shape[1]
    material_matrix = np.zeros((input_shape[2], layer_size))
    segment_matrix = np.zeros((input_shape[2], layer_size))
    is_rigid = np.zeros((input_shape[2], layer_size))
    for k in range(input_shape[2]):
        material_layer = np.zeros(layer_size)
        segment_layer = np.zeros(layer_size)
        is_rigid_layer = np.zeros(layer_size)
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                material_layer[j * input_shape[0] + i] = voxels[i, j, k]
                segment_layer[j * input_shape[0] + i] = segments[i, j, k]
                is_rigid_layer[j * input_shape[0] + i] = int(voxels[i, j, k] == rigid_material)
        material_matrix[k] = material_layer
        segment_matrix[k] = segment_layer
        is_rigid[k] = is_rigid_layer
    return material_matrix.astype(int), segment_matrix.astype(int), is_rigid.astype(int)


def generate_constraints(joint2edges, joint_pos, edges, segments, voxel_size):
    constraints = []
    rotation_idx = 0
    for joint_idx, edge_idx in joint2edges.items():
        n_edges = len(edge_idx)
        if n_edges == 2:
            if np.sum(segments == edge_idx[0] + 1) > 0 and np.sum(segments == edge_idx[1] + 1) > 0:
                anchor = joint_pos[joint_idx] * voxel_size
                v_edge_0 = np.array(edges[edge_idx[0]][1] - edges[edge_idx[0]][0])
                v_edge_1 = np.array(edges[edge_idx[1]][1] - edges[edge_idx[1]][0])
                hinge_axis = np.cross(v_edge_0, v_edge_1)
                if np.linalg.norm(hinge_axis) != 0:
                    hinge_axis /= np.linalg.norm(hinge_axis)
                    constraints.append(
                        hinge_constraint.format(edge_idx[0] + 1,
                                                anchor[0], anchor[1], anchor[2],
                                                edge_idx[1] + 1,
                                                anchor[0], anchor[1], anchor[2],
                                                hinge_axis[0], hinge_axis[1], hinge_axis[2],
                                                -hinge_axis[0], -hinge_axis[1], -hinge_axis[2],
                                                rotation_idx
                                                ))
                    rotation_idx += 1
                else:
                    constraints.append(
                        ball_and_socket_constraint.format(edge_idx[0] + 1,
                                                          anchor[0], anchor[1], anchor[2],
                                                          edge_idx[1] + 1,
                                                          anchor[0], anchor[1], anchor[2]))
    return constraints, rotation_idx


def many_bones_one_joint(joint2edges, segments):
    e2seg = {}
    for joint_idx, edge_idx in joint2edges.items():
        n_edges = len(edge_idx)
        if n_edges > 2:
            segment_id = edge_idx[0] + 1
            for i in edge_idx[1:]:
                segments[segments == i + 1] = segment_id


def fix_order(bot, shift=1):
    joint_pos = np.roll(bot.joint_positions, shift=shift, axis=-1)
    bones = np.roll(bot.bones, shift=shift, axis=-1)
    order = tuple(np.roll([0, 1, 2], shift=shift))
    voxels = bot.voxels.transpose(*order)
    segments = bot.segments.transpose(*order)
    return joint_pos, bones, voxels, segments


def export_to_rsc(shape, material_id, segment_id, segment_type, constraints, num_rotation_signals):
    material = "".join([layer.format(', '.join(map(str, m_layer.tolist()))) + '\n' for m_layer in material_id])
    segment = "".join([layer.format(', '.join(map(str, s_layer.tolist()))) + '\n' for s_layer in segment_id])
    is_rig = "".join([layer.format(', '.join(map(str, r_layer.tolist()))) + '\n' for r_layer in segment_type])
    constraints = "".join(constraints)
    return robot_rsc.format(shape, material, segment, is_rig, constraints, num_rotation_signals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='rsc generator',
        description='Generate rise config file with given robot'
    )
    parser.add_argument('filename')
    parser.add_argument('--n_voxels', type=int, default=2e4)
    parser.add_argument('--bone_radius', type=int, default=1)
    parser.add_argument('--shift', type=int, default=1)
    args = parser.parse_args()

    config_path = f"data/config/{args.filename}.rsc"

    voxel_size = 0.01
    bot = Robot(args.filename, args.n_voxels, args.bone_radius)
    joint_pos, bones, voxels, segments = fix_order(bot, args.shift)
    shape = shape_template.format(voxels.shape[0], voxels.shape[1], voxels.shape[2])
    material_id, segment_id, segment_type = generate_body(voxels, segments)
    constraints, num_rotation_signals = generate_constraints(bot.joint2bones, joint_pos, bones, segments, voxel_size)
    rsc = export_to_rsc(shape, material_id, segment_id, segment_type, constraints, num_rotation_signals)
    with open(config_path, "w") as file:
        file.write(rsc)
