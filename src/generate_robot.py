import numpy as np
from pyquaternion import Quaternion
from templates import robot_rsc, hinge_constraint, layer, shape_template
import random
# from sim.env import RiseEnv
from rise import Rise, RiseFrame
np.set_printoptions(suppress=True)


class JointNode:
    def __init__(self, length, rigid_radius, soft_radius, ax, rad):
        self.length = length
        self.rigid_radius = rigid_radius
        self.soft_radius = soft_radius
        self.ax = ax
        self.rad = rad
        self.children = []
        self.parent = None
        self.end = None
        self.world_quaternion = None
        self.id = 0

    def update(self, length, rigid_radius, soft_radius, ax, rad):
        self.length = length
        self.rigid_radius = rigid_radius
        self.soft_radius = soft_radius
        self.ax = ax
        self.rad = rad

    def add_child(self, child):
        self.children.append(child)

    def set_parent(self, parent):
        self.parent = parent

    def set_end(self, p):
        self.end = p

    def set_quaternion(self, q):
        self.world_quaternion = q

    def set_id(self, i):
        self.id = i

    def __str__(self):
        parent_id = self.parent.id if self.parent else "None"
        ax_formatted = [f"{val:.2f}" for val in self.ax]
        return (f"JointNode(ID={self.id}, "
                f"\n\t Length={self.length:.2f}, "
                f"\n\t Rigid_Radius={self.rigid_radius:.2f}, "
                f"\n\t Soft_Radius={self.soft_radius:.2f},"
                f"\n\t AX={ax_formatted}, "
                f"\n\t Rad={self.rad:.2f}, "
                f"\n\t Parent_ID={parent_id})")


def decode_gene(params):
    nodes_params = []
    for param in params:
        length = 0.05 + 0.25 * param[0]
        rigid_radius = 0.01 + 0.02 * param[1]
        soft_radius = 0.01 + 0.01 * param[2]
        ax = np.array(param[3:6]) * 2 - 1
        ax /= np.linalg.norm(ax)
        rad = (0.25 + 0.25 * param[6]) * np.pi
        nodes_params.append((length, rigid_radius, soft_radius, ax, rad))
    temp = list(nodes_params[0])
    temp[0] = 0
    temp[1] = 0.05
    nodes_params[0] = tuple(temp)
    return nodes_params


def update_nodes(nodes, params):
    for i, param in params:
        nodes[i].update(*param)
    return nodes


def initialize_topology_tree(nodes, max_children):
    available_parents = [nodes[0]]
    for node in nodes[1:]:
        if not available_parents:
            break
        parent = np.random.choice(available_parents)
        parent.add_child(node)
        node.set_parent(parent)
        if len(parent.children) >= max_children:
            available_parents.remove(parent)
        available_parents.append(node)


def generate_body(voxels, segments, expansion, rigid_material=2):
    input_shape = voxels.shape
    layer_size = input_shape[0] * input_shape[1]
    material_matrix = np.zeros((input_shape[2], layer_size))
    segment_matrix = np.zeros((input_shape[2], layer_size))
    is_rigid = np.zeros((input_shape[2], layer_size))
    expansion_matrix = np.zeros((input_shape[2], layer_size))
    for k in range(input_shape[2]):
        material_layer = np.zeros(layer_size)
        segment_layer = np.zeros(layer_size)
        is_rigid_layer = np.zeros(layer_size)
        expansion_layer = np.zeros(layer_size)
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                material_layer[j * input_shape[0] + i] = voxels[i, j, k]
                segment_layer[j * input_shape[0] + i] = segments[i, j, k]
                is_rigid_layer[j * input_shape[0] + i] = int(voxels[i, j, k] == rigid_material)
                expansion_layer[j * input_shape[0] + i] = expansion[i, j, k]
        material_matrix[k] = material_layer
        segment_matrix[k] = segment_layer
        is_rigid[k] = is_rigid_layer
        expansion_matrix[k] = expansion_layer
    return material_matrix.astype(int), segment_matrix.astype(int), is_rigid.astype(int), expansion_matrix.astype(int)


def add_constraint(constraints, anchor, segment_i, segment_j, hinge_axis, id):
    constraints.append(
        hinge_constraint.format(segment_i,
                                anchor[0], anchor[1], anchor[2],
                                segment_j,
                                anchor[0], anchor[1], anchor[2],
                                hinge_axis[0], hinge_axis[1], hinge_axis[2],
                                -hinge_axis[0], -hinge_axis[1], -hinge_axis[2],
                                id
                                ))


def export_to_rsc(size, material_id, segment_id, segment_type, expansion_sig, constraints, e_signals):
    material = "".join([layer.format(', '.join(map(str, m_layer.tolist()))) + '\n' for m_layer in material_id])
    segment = "".join([layer.format(', '.join(map(str, s_layer.tolist()))) + '\n' for s_layer in segment_id])
    is_rig = "".join([layer.format(', '.join(map(str, r_layer.tolist()))) + '\n' for r_layer in segment_type])
    expansion = "".join([layer.format(', '.join(map(str, es_layer.tolist()))) + '\n' for es_layer in expansion_sig])
    constraint_str = "".join(constraints)
    return robot_rsc.format(size, material, segment, is_rig, expansion, constraint_str, e_signals, len(constraints))


def update(voxels, segments, expansion, segment_idx,
           bone_coords, musc_coords):
    voxels[
        musc_coords[:, 0],
        musc_coords[:, 1],
        musc_coords[:, 2],
    ] += voxels[
             musc_coords[:, 0],
             musc_coords[:, 1],
             musc_coords[:, 2],
         ] == 0

    voxels[
        bone_coords[:, 0],
        bone_coords[:, 1],
        bone_coords[:, 2],
    ] = 2

    segments[
        bone_coords[:, 0],
        bone_coords[:, 1],
        bone_coords[:, 2],
    ] = segment_idx

    expansion[
        musc_coords[:, 0],
        musc_coords[:, 1],
        musc_coords[:, 2],
    ] = segment_idx - 1


def add_one_component(coords, start, end, node, dx):
    # start = np.zeros(3) if node.parent is None else node.parent.end
    if node.length == 0:
        dist = np.linalg.norm(start - coords, axis=1)
        bone_coords = coords[dist <= node.rigid_radius / dx]
        musc_coords = coords[
            np.logical_and(dist > node.rigid_radius / dx, dist <= (node.rigid_radius + node.soft_radius) / dx)]
    else:
        # end = node.end
        u = end - start
        proj_mat = np.outer(u, u) / np.dot(u, u)
        proj_ps = np.dot(proj_mat, (coords - start).T).T + start
        dist = np.linalg.norm(proj_ps - coords, axis=1)
        proj_dist = np.linalg.norm(proj_ps - start.T, axis=1) + np.linalg.norm(proj_ps - end.T, axis=1)
        in_seg = np.isclose(proj_dist, node.length / dx, rtol=1e-6)
        t = np.dot(coords - start, u) / np.dot(u, u)
        t = np.clip(t, 0, 1)
        closet_p = u * t[:, np.newaxis] + start
        dist_to_seg = np.linalg.norm(closet_p - coords, axis=1)
        in_cyl = dist_to_seg <= (node.rigid_radius + node.soft_radius) / dx
        bone_bool = np.logical_and(dist <= node.rigid_radius / dx, in_seg)
        musc_bool = np.logical_xor(in_cyl, bone_bool)
        bone_coords = coords[bone_bool]
        musc_coords = coords[musc_bool]
    return bone_coords, musc_coords


def parse(node, up, start=None):
    if start is None:
        start = node.parent.end
    if node.length == 0:
        q = Quaternion(axis=node.ax, angle=node.rad)
        v = q.rotate(up)
        end = start + v * node.rigid_radius
        
    else:
        q = node.parent.world_quaternion
        q *= Quaternion(axis=node.ax, angle=node.rad)
        v = q.rotate(up)
        end = start + v * node.length
    node.set_end(end)
    node.set_quaternion(q)


def parse_topology_tree(root, root_position=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    segment_idx = 1
    parse(root, up, root_position)
    root.set_id(segment_idx)
    queue = []
    queue += root.children
    while len(queue) != 0:
        curr = queue.pop()
        queue += curr.children
        parse(curr, up)
        segment_idx += 1
        curr.set_id(segment_idx)


def build(coords, voxels, segments, expansion, root, lb, dx, offset, root_position=np.array([0, 0, 0]), up=np.array([0,0,1])):
    constraints = []
    start, end = coords_position(root_position, lb, dx, offset / 2), coords_position(root.end, lb, dx, offset / 2)
    bone_coords, musc_coords = add_one_component(coords, start, end, root, dx)
    update(voxels, segments, expansion, root.id, bone_coords, musc_coords)
    queue = []
    queue += root.children
    rid = 0
    while len(queue) != 0:
        curr = queue.pop()
        queue += curr.children
        start, end = coords_position(curr.parent.end, lb, dx, offset / 2), coords_position(curr.end, lb, dx, offset / 2)
        bone_coords, musc_coords = add_one_component(coords, start, end, curr, dx)
        update(voxels, segments, expansion, curr.id, bone_coords, musc_coords)
        h_a = np.cross(curr.parent.world_quaternion.rotate(up), curr.world_quaternion.rotate(up))
        h_a /= np.linalg.norm(h_a)
        add_constraint(constraints, start * dx, curr.parent.id, curr.id, h_a, rid)
        rid += 1
    return constraints


def coords_position(p, lower_bd, dx, offset):
    return np.array((p - lower_bd) / dx) + offset


def initialize_matrices(joint_positions, root, dx=.01, offset=np.array([20, 20, 10])):
    lower_bd = np.min(joint_positions, axis=0)
    upper_bd = np.max(joint_positions, axis=0)
    size = tuple(np.ceil((upper_bd - lower_bd) / dx + offset).astype(int))
    voxels = np.zeros(size)
    segments = np.zeros(size)
    expansion = np.zeros(size) - 1
    coords = np.array(list(np.ndindex(size)))
    constraints = build(coords, voxels, segments, expansion, root, lower_bd, dx, offset)
    return size, voxels, segments, expansion, constraints


if __name__ == '__main__':
    num_nodes = 10
    np.random.seed(111)
    params = np.random.rand(num_nodes, 7)
    nodes_params = decode_gene(params)
    nodes = [JointNode(*params) for params in nodes_params]
    initialize_topology_tree(nodes, 2)
    parse_topology_tree(nodes[0])
    # print('\n'.join(str(node) for node in nodes))
    joint_positions = np.zeros((len(nodes) + 1, 3))
    joint_positions[1:] = np.array([node.end for node in nodes])
    # print(joint_positions)
    size, voxels, segments, expansion, constraints = initialize_matrices(joint_positions, nodes[0])
    shape = shape_template.format(size[0], size[1], size[2])
    material_id, segment_id, segment_type, expansion_sig = generate_body(voxels, segments, expansion)
    rsc = export_to_rsc(shape, material_id, segment_id, segment_type, expansion_sig, constraints,
                    int(np.max(expansion)))
    with open('../data/config/generated.rsc', 'w') as f:
        f.write(rsc)

    with open("../data/env.rsc", "r") as f:
        env_config = f.read()

    rise = Rise(devices=[0])

    configs = []
    robot_configs = []

    configs.append([env_config, rsc])
    robot_configs.append(rsc)

    result = rise.run_sims(
        configs,
        [],
        record_buffer_size=1,
        save_result=True,
        save_record=True,
        log_level="debug"
    )
    #
    # env = RiseEnv(
    #     devices=[0],
    #     env_config=env_config,
    #     voxel_size=0.01,
    #     material_num=3,
    #     material_is_rigid=segments == 2,
    #     voxel_grid_size=size,
    #     result_path="../data/result/generated.result",
    #     record_path="../data/result/generated.history"
    # )
    #
    # env.run_sims(
    #     generation=0,
    #     robots=,
    #     controller = None,
    #     builder_kwargs = None,
    #     fitness_critique_observe_frames = 10,
    #     fitness_critique_observe_interval= 0.1,
    #     save_record = False,
    #     record_buffer_size = 100,
    # )
