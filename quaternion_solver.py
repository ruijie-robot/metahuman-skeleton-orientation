import numpy as np
from typing import List, Tuple, Optional
from metahuman_skeleton import MetahumanSkeleton

class QuaternionSolver:
    """Converts world coordinates to local quaternions for metahuman skeleton"""
    
    def __init__(self):
        self.skeleton = MetahumanSkeleton()
    
    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        """Normalize a vector"""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0])  # Default to Z-up if zero vector
        return v / norm
    
    @staticmethod
    def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        trace = np.trace(matrix)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 2] - matrix[2, 0]) / s
            z = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        """Get quaternion conjugate"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def compute_bone_orientation(self, bone_pos: np.ndarray, child_pos: Optional[np.ndarray] = None, 
                               parent_orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute bone orientation from position and child position"""
        if child_pos is None:
            # End effector - use identity quaternion
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        # Bone direction vector
        bone_direction = self.normalize_vector(child_pos - bone_pos)
        
        # Default bone direction (assuming Y-axis)
        default_direction = np.array([0.0, 1.0, 0.0])
        
        # Compute rotation to align default direction with bone direction
        dot_product = np.dot(default_direction, bone_direction)
        
        if abs(dot_product + 1.0) < 1e-6:
            # 180-degree rotation
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(default_direction, perp)) > 0.9:
                perp = np.array([0.0, 0.0, 1.0])
            quaternion = np.array([0.0, perp[0], perp[1], perp[2]])
        elif abs(dot_product - 1.0) < 1e-6:
            # No rotation needed
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Normal rotation
            cross_product = np.cross(default_direction, bone_direction)
            w = 1.0 + dot_product
            quaternion = np.array([w, cross_product[0], cross_product[1], cross_product[2]])
            quaternion = quaternion / np.linalg.norm(quaternion)
        
        return quaternion
    
    def world_to_local_quaternions(self, world_positions: np.ndarray) -> np.ndarray:
        """
        Convert world positions to local quaternions
        
        Args:
            world_positions: Array of shape (68, 3) containing world coordinates for all bones
            
        Returns:
            Array of shape (68, 4) containing local quaternions (w, x, y, z) for all bones
        """
        if world_positions.shape != (68, 3):
            raise ValueError(f"Expected shape (68, 3), got {world_positions.shape}")
        
        local_quaternions = np.zeros((68, 4))
        world_quaternions = np.zeros((68, 4))
        
        # Process bones in hierarchical order
        for bone_idx in range(68):
            parent_idx = self.skeleton.parent_indices[bone_idx]
            children = self.skeleton.get_children(bone_idx)
            
            bone_pos = world_positions[bone_idx]
            
            # Find primary child for bone orientation
            child_pos = None
            if children:
                # Use first child as primary direction
                child_pos = world_positions[children[0]]
            
            # Compute world orientation for this bone
            world_quat = self.compute_bone_orientation(bone_pos, child_pos)
            world_quaternions[bone_idx] = world_quat
            
            # Convert to local space relative to parent
            if parent_idx == -1:
                # Root bone - world quaternion is local quaternion
                local_quaternions[bone_idx] = world_quat
            else:
                # Child bone - compute relative to parent
                parent_world_quat = world_quaternions[parent_idx]
                parent_conjugate = self.quaternion_conjugate(parent_world_quat)
                local_quaternions[bone_idx] = self.quaternion_multiply(parent_conjugate, world_quat)
        
        return local_quaternions
    
    def process_animation_sequence(self, animation_data: np.ndarray) -> np.ndarray:
        """
        Process entire animation sequence
        
        Args:
            animation_data: Array of shape (num_frames, 68, 3) containing world coordinates
            
        Returns:
            Array of shape (num_frames, 68, 4) containing local quaternions
        """
        num_frames = animation_data.shape[0]
        if animation_data.shape[1:] != (68, 3):
            raise ValueError(f"Expected shape (num_frames, 68, 3), got {animation_data.shape}")
        
        result = np.zeros((num_frames, 68, 4))
        
        for frame_idx in range(num_frames):
            result[frame_idx] = self.world_to_local_quaternions(animation_data[frame_idx])
        
        return result