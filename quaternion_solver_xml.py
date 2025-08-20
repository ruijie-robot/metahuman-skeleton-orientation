import numpy as np
from typing import List, Tuple, Optional
from urdf_parser import URDFParser, URDFJoint

class QuaternionSolverXML:
    """从URDF XML文件加载骨架并计算局部四元数"""
    
    def __init__(self, urdf_file_path: str = "metahuman.urdf"):
        """
        初始化四元数求解器
        
        Args:
            urdf_file_path: URDF XML文件路径
        """
        self.urdf_parser = URDFParser(urdf_file_path)
        self.links = self.urdf_parser.links
        self.joints = self.urdf_parser.joints
        
        print(f"Loaded skeleton from {urdf_file_path}")
        print(f"  Links: {len(self.links)}")
        print(f"  Joints: {len(self.joints)}")
        
        # 验证结构
        if not self.urdf_parser.validate_structure():
            raise ValueError("URDF structure validation failed")
    
    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        """Normalize a vector"""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0])  # Default to Z-up if zero vector
        return v / norm
    
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
    
    def compute_joint_orientation(self, joint: URDFJoint, parent_pos: np.ndarray, child_pos: np.ndarray) -> np.ndarray:
        """计算关节相对于T-pose的朝向"""
        # 当前方向向量 (从parent到child)
        current_direction = self.normalize_vector(child_pos - parent_pos)
        
        # 获取该关节在T-pose中的初始方向
        if joint.name == None:
            initial_direction = np.array([0.0, 1.0, 0.0])
        else:
            initial_direction = joint.tpose_direction
        
        # 计算从T-pose到当前朝向的旋转
        return self.quaternion_from_vectors_standard(initial_direction, current_direction)
    
    def world_to_local_quaternions(self, world_positions: np.ndarray) -> np.ndarray:
        """
        将世界坐标转换为关节连接的局部四元数
        
        Args:
            world_positions: 形状为 (68, 3) 的数组，包含所有骨骼的世界坐标
            
        Returns:
            形状为 (67, 4) 的数组，包含关节连接的局部四元数 (w, x, y, z)
        """
        expected_links = len(self.links)
        if world_positions.shape != (expected_links, 3):
            raise ValueError(f"Expected shape ({expected_links}, 3), got {world_positions.shape}")
        
        num_joints = len(self.joints)
        local_quaternions = np.zeros((num_joints+1, 4))
        
        # 首先处理root骨骼点
        root_pos = world_positions[0]
        # 如果root_pos的坐标系是(0,0,0), 那么orientation使用(1, 0, 0, 0)
        if np.allclose(root_pos, np.zeros(3)):
            root_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # 反之，origin_pos = (0,0,0), 计算local_quat = self.compute_joint_orientation(joint, origin_pos, root_pos)
            origin_pos = np.zeros(3)
            # 这里假设root joint为第一个joint
            root_orientation = self.compute_joint_orientation(None, origin_pos, root_pos)
        local_quaternions[0] = root_orientation

        # 按顺序处理关节
        for joint_idx, joint in enumerate(self.joints):
            parent_pos = world_positions[joint.parent_index]
            child_pos = world_positions[joint.child_index]
                
            # 计算该关节相对于T-pose的世界朝向
            local_quat = self.compute_joint_orientation(joint, parent_pos, child_pos)
            local_quaternions[joint_idx+1] = local_quat
        
        return local_quaternions
    
    def process_animation_sequence(self, animation_data: np.ndarray) -> np.ndarray:
        """
        处理整个动画序列
        
        Args:
            animation_data: 形状为 (num_frames, 68, 3) 的数组，包含世界坐标
            
        Returns:
            形状为 (num_frames, 67, 4) 的数组，包含关节连接的局部四元数
        """
        num_frames = animation_data.shape[0]
        expected_links = len(self.links)
        
        if animation_data.shape[1:] != (expected_links, 3):
            raise ValueError(f"Expected shape (num_frames, {expected_links}, 3), got {animation_data.shape}")
        
        num_joints = len(self.joints)
        result = np.zeros((num_frames, num_joints, 4))
        
        for frame_idx in range(num_frames):
            result[frame_idx] = self.world_to_local_quaternions(animation_data[frame_idx])
        
        return result
    
    def get_joint_info(self, joint_name: str) -> Optional[URDFJoint]:
        """获取关节信息"""
        return self.urdf_parser.get_joint_by_name(joint_name)
    
    def get_link_info(self, link_name: str) -> Optional:
        """获取链接信息"""
        return self.urdf_parser.get_link_by_name(link_name)
    
    def print_skeleton_summary(self):
        """打印骨架摘要"""
        self.urdf_parser.print_summary()
        
    def get_joint_names(self) -> List[str]:
        """获取所有关节名称"""
        return [joint.name for joint in self.joints]
    
    def get_link_names(self) -> List[str]:
        """获取所有链接名称"""
        return [link.name for link in self.links]