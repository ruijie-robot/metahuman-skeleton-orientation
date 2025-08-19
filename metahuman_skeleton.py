import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class BoneHierarchy:
    """Metahuman skeleton bone hierarchy definition"""
    name: str
    parent_index: int
    children_indices: List[int]

class MetahumanSkeleton:
    """Metahuman skeleton with 67 main bones plus root"""
    
    def __init__(self):
        self.bone_names = [
            "root",  # 0
            "pelvis",  # 1
            "spine_01", "spine_02", "spine_03",  # 2-4
            "neck_01", "head",  # 5-6
            "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",  # 7-10
            "thumb_01_l", "thumb_02_l", "thumb_03_l",  # 11-13
            "index_01_l", "index_02_l", "index_03_l",  # 14-16
            "middle_01_l", "middle_02_l", "middle_03_l",  # 17-19
            "ring_01_l", "ring_02_l", "ring_03_l",  # 20-22
            "pinky_01_l", "pinky_02_l", "pinky_03_l",  # 23-25
            "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",  # 26-29
            "thumb_01_r", "thumb_02_r", "thumb_03_r",  # 30-32
            "index_01_r", "index_02_r", "index_03_r",  # 33-35
            "middle_01_r", "middle_02_r", "middle_03_r",  # 36-38
            "ring_01_r", "ring_02_r", "ring_03_r",  # 39-41
            "pinky_01_r", "pinky_02_r", "pinky_03_r",  # 42-44
            "thigh_l", "calf_l", "foot_l", "ball_l",  # 45-48
            "thigh_r", "calf_r", "foot_r", "ball_r",  # 49-52
            "ik_foot_root", "ik_foot_l", "ik_foot_r",  # 53-55
            "ik_hand_root", "ik_hand_gun", "ik_hand_l", "ik_hand_r",  # 56-59
            "jaw", "eye_l", "eye_r",  # 60-62
            "breast_l", "breast_r",  # 63-64
            "twist_01_thigh_l", "twist_01_thigh_r",  # 65-66
            "twist_01_upperarm_l"  # 67
        ]
        
        # Parent-child relationships (parent index for each bone)
        self.parent_indices = [
            -1,  # root has no parent，root代表了世界坐标系的原点，
            0,   # pelvis -> root
            1,   # spine_01 -> pelvis
            2,   # spine_02 -> spine_01
            3,   # spine_03 -> spine_02
            4,   # neck_01 -> spine_03
            5,   # head -> neck_01
            4,   # clavicle_l -> spine_03
            7,   # upperarm_l -> clavicle_l
            8,   # lowerarm_l -> upperarm_l
            9,   # hand_l -> lowerarm_l
            10,  # thumb_01_l -> hand_l
            11,  # thumb_02_l -> thumb_01_l
            12,  # thumb_03_l -> thumb_02_l
            10,  # index_01_l -> hand_l
            14,  # index_02_l -> index_01_l
            15,  # index_03_l -> index_02_l
            10,  # middle_01_l -> hand_l
            17,  # middle_02_l -> middle_01_l
            18,  # middle_03_l -> middle_02_l
            10,  # ring_01_l -> hand_l
            20,  # ring_02_l -> ring_01_l
            21,  # ring_03_l -> ring_02_l
            10,  # pinky_01_l -> hand_l
            23,  # pinky_02_l -> pinky_01_l
            24,  # pinky_03_l -> pinky_02_l
            4,   # clavicle_r -> spine_03
            26,  # upperarm_r -> clavicle_r
            27,  # lowerarm_r -> upperarm_r
            28,  # hand_r -> lowerarm_r
            29,  # thumb_01_r -> hand_r
            30,  # thumb_02_r -> thumb_01_r
            31,  # thumb_03_r -> thumb_02_r
            29,  # index_01_r -> hand_r
            33,  # index_02_r -> index_01_r
            34,  # index_03_r -> index_02_r
            29,  # middle_01_r -> hand_r
            36,  # middle_02_r -> middle_01_r
            37,  # middle_03_r -> middle_02_r
            29,  # ring_01_r -> hand_r
            39,  # ring_02_r -> ring_01_r
            40,  # ring_03_r -> ring_02_r
            29,  # pinky_01_r -> hand_r
            42,  # pinky_02_r -> pinky_01_r
            43,  # pinky_03_r -> pinky_02_r
            1,   # thigh_l -> pelvis
            45,  # calf_l -> thigh_l
            46,  # foot_l -> calf_l
            47,  # ball_l -> foot_l
            1,   # thigh_r -> pelvis
            49,  # calf_r -> thigh_r
            50,  # foot_r -> calf_r
            51,  # ball_r -> foot_r
            0,   # ik_foot_root -> root
            53,  # ik_foot_l -> ik_foot_root
            53,  # ik_foot_r -> ik_foot_root
            0,   # ik_hand_root -> root
            56,  # ik_hand_gun -> ik_hand_root
            56,  # ik_hand_l -> ik_hand_root
            56,  # ik_hand_r -> ik_hand_root
            6,   # jaw (60) -> head (6)
            6,   # eye_l (61) -> head (6)
            6,   # eye_r (62) -> head (6)
            4,   # breast_l -> spine_03
            4,   # breast_r -> spine_03
            45,  # twist_01_thigh_l -> thigh_l
            49,  # twist_01_thigh_r -> thigh_r
            8    # twist_01_upperarm_l -> upperarm_l
        ]
        
        assert len(self.bone_names) == 68
        assert len(self.parent_indices) == 68
    
    def get_children(self, bone_index: int) -> List[int]:
        """Get children indices for a given bone"""
        children = []
        for i, parent_idx in enumerate(self.parent_indices):
            if parent_idx == bone_index:
                children.append(i)
        return children
    
    def get_tpose_bone_direction(self, bone_index: int) -> np.ndarray:
        """Get the initial bone direction in Unity T-pose coordinate system"""
        bone_name = self.bone_names[bone_index]
        
        # Unity T-pose coordinate system: X-right, Y-up, Z-forward
        
        if "spine" in bone_name or "neck" in bone_name:
            return np.array([0.0, 1.0, 0.0])  # Y-up (向上)
            
        elif "clavicle" in bone_name:
            if "_l" in bone_name:
                return np.array([-1.0, 0.0, 0.0])  # X负方向 (向左)
            else:
                return np.array([1.0, 0.0, 0.0])   # X正方向 (向右)
                
        elif "upperarm" in bone_name or "lowerarm" in bone_name or "hand" in bone_name:
            if "_l" in bone_name:
                return np.array([-1.0, 0.0, 0.0])  # X负方向 (左臂水平向左)
            else:
                return np.array([1.0, 0.0, 0.0])   # X正方向 (右臂水平向右)
                
        elif "thigh" in bone_name or "calf" in bone_name:
            return np.array([0.0, -1.0, 0.0])     # Y负方向 (腿部向下)
            
        elif "foot" in bone_name or "ball" in bone_name:
            return np.array([0.0, 0.0, 1.0])      # Z正方向 (脚趾向前)
            
        elif "thumb" in bone_name:
            if "_l" in bone_name:
                return np.array([0.0, 0.0, 1.0])   # Z正方向 (拇指向前)
            else:
                return np.array([0.0, 0.0, 1.0])   # Z正方向
                
        elif "index" in bone_name or "middle" in bone_name or "ring" in bone_name or "pinky" in bone_name:
            if "_l" in bone_name:
                return np.array([-1.0, 0.0, 0.0]) # X负方向 (左手手指向左延伸)
            else:
                return np.array([1.0, 0.0, 0.0])  # X正方向 (右手手指向右延伸)
                
        elif "head" in bone_name:
            return np.array([0.0, 1.0, 0.0])      # Y正方向 (头部向上)
            
        elif "jaw" in bone_name:
            return np.array([0.0, -1.0, 0.0])     # Y负方向 (下巴向下)
            
        elif "eye" in bone_name:
            return np.array([0.0, 0.0, 1.0])      # Z正方向 (眼睛向前)
            
        elif "breast" in bone_name:
            return np.array([0.0, 0.0, 1.0])      # Z正方向 (胸部向前)
            
        elif "twist" in bone_name:
            # 扭转骨骼跟随父骨骼方向
            parent_idx = self.parent_indices[bone_index]
            if parent_idx >= 0:
                return self.get_tpose_bone_direction(parent_idx)
            else:
                return np.array([0.0, 1.0, 0.0])
                
        elif "ik_" in bone_name:
            return np.array([0.0, 1.0, 0.0])      # IK骨骼默认向上
            
        else:
            # 默认方向
            return np.array([0.0, 1.0, 0.0])      # Y正方向 (默认向上)