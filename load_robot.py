import trimesh
import numpy as np


class Robot:
    def __init__(self, filename, n_voxels=1e4, radius=2):
        surface_path = f"data/surface/{filename}.obj"
        bone_path = f"data/bone/{filename}.txt"

        surface = self.load_surface(surface_path)
        bones, joint_positions, self.joint2bones = self.load_bones(bone_path)
        self.dx = self._find_optimal_dx(surface, n_voxels)
        self.vox_surface = surface.voxelized(self.dx)
        filled_vox = self.vox_surface.fill()
        self.voxels = np.array(filled_vox.matrix).astype(int)
        self.segments = np.zeros_like(self.voxels)
        bounds = self.vox_surface.as_boxes().bounds
        self.bones = self._update_coords(bones, self.dx, bounds[None , 0])
        self.joint_positions = self._update_coords(joint_positions, self.dx, bounds[None , 0])
        self.voxels, self.segments = self.add_bones(self.voxels, self.segments, self.bones, radius)

    def load_surface(self, path):
        return trimesh.load(path, force="mesh")

    def load_bones(self, path):
        joint2idx = {}
        joint_postions = []
        bones = []
        joint2bones = {}

        with open(path, "r") as f:
            lines = f.readlines()
        edge_idx = 0
        for line in lines:
            if line.startswith("joints"):
                tokens = line.split()
                joint_name = tokens[1]
                joint2idx[joint_name] = len(joint2idx)
                joint_postions.append(np.array(tokens[-3:], dtype=float))
            if line.startswith("hier"):
                tokens = line.split()
                start, end = joint2idx[tokens[1]], joint2idx[tokens[2]]
                bones.append((joint_postions[start], joint_postions[end]))
                joint2bones.setdefault(start, []).append(edge_idx)
                joint2bones.setdefault(end, []).append(edge_idx)
                edge_idx += 1
        return np.array(bones), np.array(joint_postions), joint2bones
    
    def add_one_bone(self, coords, bone, radius):
        dir = bone[1] - bone[0]
        proj_mat = np.outer(dir, dir) / np.sum(dir ** 2)
        proj_ps = np.dot(proj_mat, (coords - bone[0]).T).T + bone[0]
        distance = np.linalg.norm(proj_ps - coords, axis=1)
        bone_length = np.linalg.norm(dir)
        proj_dist = np.linalg.norm(proj_ps - bone[0].T, axis=1)
        is_on_segment = np.logical_and(proj_dist > 0, proj_dist < bone_length)
        return coords[np.logical_and(distance <= radius, is_on_segment)]

    def add_bones(self, voxels, segments, bones, radius=2, material=1):
        coords = np.array(list(np.ndindex(voxels.shape)))
        for i, bone in enumerate(bones):
            segment_idx = i + 1
            bone_coords = self.add_one_bone(coords, bone, radius)
            voxels[
                bone_coords[:, 0],
                bone_coords[:, 1],
                bone_coords[:, 2],
            ] = material

            segments[
                bone_coords[:, 0],
                bone_coords[:, 1],
                bone_coords[:, 2],
            ] = segment_idx
        return voxels, segments

    def _update_coords(self, coords, dx, lower_bound):
        return (coords - lower_bound) / dx

    def _find_optimal_dx(self, mesh, n_voxels, min_dx=1e-6, max_dx=.1, tolerance=1e-6):
        checked = max_dx
        while (max_dx - min_dx) > tolerance:
            mid_dx = (max_dx + min_dx) / 2
            voxel_grid = mesh.voxelized(mid_dx)
            voxel_grid = voxel_grid.fill()
            curr_voxels = np.sum(voxel_grid.matrix)
            if curr_voxels > n_voxels:
                min_dx = mid_dx
            elif curr_voxels < n_voxels:
                max_dx = mid_dx
                checked = min(checked, mid_dx)
            else:
                return mid_dx
        return checked
    
    def export_to_box_mesh():
        pass
