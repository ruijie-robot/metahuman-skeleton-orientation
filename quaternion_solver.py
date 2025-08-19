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
    
    def quaternion_from_vectors_standard(self, vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
        """
        Standard quaternion calculation method
        q = [cos(θ/2), u_x * sin(θ/2), u_y * sin(θ/2), u_z * sin(θ/2)]
        where axis = (u_x, u_y, u_z) = normalize(a × b)
        """
        vec_from = self.normalize_vector(vec_from)
        vec_to = self.normalize_vector(vec_to)
        
        dot_product = np.dot(vec_from, vec_to)
        
        if abs(dot_product + 1.0) < 1e-6:
            # 180-degree rotation
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(vec_from, perp)) > 0.9:
                perp = np.array([0.0, 0.0, 1.0])
            return np.array([0.0, perp[0], perp[1], perp[2]])
        elif abs(dot_product - 1.0) < 1e-6:
            # No rotation needed
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Calculate rotation angle and axis
            theta = np.arccos(np.clip(dot_product, -1.0, 1.0))
            cross_product = np.cross(vec_from, vec_to)
            axis = self.normalize_vector(cross_product)
            
            # Standard quaternion formula
            cos_half_theta = np.cos(theta / 2.0)
            sin_half_theta = np.sin(theta / 2.0)
            
            quaternion = np.array([
                cos_half_theta,
                axis[0] * sin_half_theta,
                axis[1] * sin_half_theta,
                axis[2] * sin_half_theta
            ])
            
            return quaternion
    
    def quaternion_from_vectors_optimized(self, vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
        """
        Optimized quaternion calculation method (avoids trigonometric functions)
        q' = [2*cos²(θ/2), 2*cos(θ/2)*sin(θ/2) * axis]
           = [1 + cos(θ), sin(θ) * axis]
           = [1 + dot_product, cross_product]
        """
        vec_from = self.normalize_vector(vec_from)
        vec_to = self.normalize_vector(vec_to)
        
        dot_product = np.dot(vec_from, vec_to)
        
        if abs(dot_product + 1.0) < 1e-6:
            # 180-degree rotation
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(vec_from, perp)) > 0.9:
                perp = np.array([0.0, 0.0, 1.0])
            return np.array([0.0, perp[0], perp[1], perp[2]])
        elif abs(dot_product - 1.0) < 1e-6:
            # No rotation needed
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Optimized calculation
            cross_product = np.cross(vec_from, vec_to)
            w = 1.0 + dot_product  # = 2*cos²(θ/2)
            quaternion = np.array([w, cross_product[0], cross_product[1], cross_product[2]])
            quaternion = quaternion / np.linalg.norm(quaternion)
            
            return quaternion

    def compute_bone_orientation(self, bone_index: int, bone_pos: np.ndarray, 
                               child_index: int, child_pos: np.ndarray) -> np.ndarray:
        """Compute bone orientation from parent-child positions relative to T-pose"""
        # Current bone direction vector (from parent to child)
        current_bone_direction = self.normalize_vector(child_pos - bone_pos)
        
        # Get the initial T-pose direction for this parent→child connection
        initial_bone_direction = self.skeleton.get_tpose_parent_to_child_direction(bone_index, child_index)
        
        # Use standard method to compute rotation from T-pose to current orientation
        return self.quaternion_from_vectors_standard(initial_bone_direction, current_bone_direction)
    
    def world_to_local_quaternions(self, world_positions: np.ndarray) -> np.ndarray:
        """
        Convert world positions to local quaternions for bone connections
        
        Args:
            world_positions: Array of shape (68, 3) containing world coordinates for all bones
            
        Returns:
            Array of shape (67, 4) containing local quaternions (w, x, y, z) for bone connections
            Index i represents the quaternion for connection from parent_indices[i+1] to bone i+1
        """
        if world_positions.shape != (68, 3):
            raise ValueError(f"Expected shape (68, 3), got {world_positions.shape}")
        
        # 67 connections (bone 1-67, since bone 0 is root with no parent)
        local_quaternions = np.zeros((67, 4))
        world_quaternions = np.zeros((67, 4))
        
        # Process bone connections (skip root bone index 0)
        for bone_idx in range(1, 68):
            parent_idx = self.skeleton.parent_indices[bone_idx]
            connection_idx = bone_idx - 1  # Map bone index to connection index
            
            # Get positions
            parent_pos = world_positions[parent_idx]
            child_pos = world_positions[bone_idx]
            
            # Compute world orientation for this connection relative to T-pose
            world_quat = self.compute_bone_orientation(parent_idx, parent_pos, bone_idx, child_pos)
            world_quaternions[connection_idx] = world_quat
            
            # Convert to local space relative to parent connection
            if parent_idx == 0:
                # Direct child of root - world quaternion is local quaternion
                local_quaternions[connection_idx] = world_quat
            else:
                # Child of non-root bone - compute relative to parent connection
                parent_connection_idx = parent_idx - 1
                if parent_connection_idx >= 0:
                    parent_world_quat = world_quaternions[parent_connection_idx]
                    parent_conjugate = self.quaternion_conjugate(parent_world_quat)
                    local_quaternions[connection_idx] = self.quaternion_multiply(parent_conjugate, world_quat)
                else:
                    local_quaternions[connection_idx] = world_quat
        
        return local_quaternions
    
    def process_animation_sequence(self, animation_data: np.ndarray) -> np.ndarray:
        """
        Process entire animation sequence
        
        Args:
            animation_data: Array of shape (num_frames, 68, 3) containing world coordinates
            
        Returns:
            Array of shape (num_frames, 67, 4) containing local quaternions for bone connections
        """
        num_frames = animation_data.shape[0]
        if animation_data.shape[1:] != (68, 3):
            raise ValueError(f"Expected shape (num_frames, 68, 3), got {animation_data.shape}")
        
        result = np.zeros((num_frames, 67, 4))
        
        for frame_idx in range(num_frames):
            result[frame_idx] = self.world_to_local_quaternions(animation_data[frame_idx])
        
        return result