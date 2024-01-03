import argparse
import numpy as np
from itertools import combinations
from templates import *
from load_robot import Robot


def generate_body(voxels, segments, rigid_material=2):
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

def generate_constraints(joint2edges, joint_pos, edges):
        constraints = []
        for joint_idx, edge_idx in joint2edges.items():
            n_edges = len(edge_idx)
            anchor = joint_pos[joint_idx]
            if n_edges == 2:
                v_edge_0 = np.array(edges[edge_idx[0]][1] - edges[edge_idx[0]][0])
                v_edge_1 = np.array(edges[edge_idx[1]][1] - edges[edge_idx[1]][0])
                hinge_axis = np.cross(v_edge_0, v_edge_1)
                if np.any(np.isnan(hinge_axis)):
                    hinge_axis = np.array([1,0,0])
                if np.linalg.norm(hinge_axis) != 0:
                    hinge_axis /= np.linalg.norm(hinge_axis)

                constraints.append(
                    hinge_constraint.format(edge_idx[0]+1,
                                            anchor[0], anchor[1], anchor[2],
                                            edge_idx[1]+1,
                                            anchor[0], anchor[1], anchor[2],
                                            hinge_axis[0],hinge_axis[1],hinge_axis[2],
                                            -hinge_axis[0],-hinge_axis[1],-hinge_axis[2]
                                            ))
            else:
                for edge_pair in list(combinations(edge_idx, 2)):
                    constraints.append(
                        fixed_constraint.format(edge_pair[0]+1,
                                                anchor[0], anchor[1], anchor[2],
                                                edge_pair[1]+1,
                                                anchor[0], anchor[1], anchor[2]))
        return constraints

def export_to_rsc(shape, material_id, segment_id, segment_type, constraints):
    material = "".join([layer.format(', '.join(map(str, m_layer.tolist()))) + '\n' for m_layer in material_id])
    segment = "".join([layer.format(', '.join(map(str, s_layer.tolist()))) + '\n' for s_layer in segment_id])
    is_rig = "".join([layer.format(', '.join(map(str, r_layer.tolist()))) + '\n' for r_layer in segment_type])
    constraints = "".join(constraints)
    return robot_rsc.format(shape, material, segment, is_rig, constraints)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='rsc generator',
        description='Generate rise config file with given robot'
    )
    parser.add_argument('filename')
    parser.add_argument('--n_voxels', type=int, default=1e4)
    parser.add_argument('--bone_radius', type=int, default=2)
    args = parser.parse_args()
    
    config_path = f"data/config/{args.filename}.rsc"
    bot = Robot(args.filename, args.n_voxels, args.bone_radius)
    shape = shape_template.format(bot.voxels.shape[0], bot.voxels.shape[1], bot.voxels.shape[2])
    material_id, segment_id, segment_type = generate_body(bot.voxels, bot.segments)
    constraints = generate_constraints(bot.joint2bones, bot.joint_positions, bot.bones)
    rsc = export_to_rsc(shape, material_id, segment_id, segment_type, constraints)
    with open(config_path, "w") as file:
        file.write(rsc)