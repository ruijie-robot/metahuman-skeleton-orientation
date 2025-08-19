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
    
    def get_tpose_parent_to_child_direction(self, parent_index: int, child_index: int) -> np.ndarray:
        """
        Get the initial direction from parent bone to child bone in Unity T-pose
        Unity T-pose coordinate system: X-right, Y-up, Z-forward
        """
        if parent_index < 0 or child_index < 0 or parent_index >= len(self.bone_names) or child_index >= len(self.bone_names):
            return np.array([0.0, 1.0, 0.0])  # Default direction
            
        parent_name = self.bone_names[parent_index]
        child_name = self.bone_names[child_index]
        
        # Define T-pose directions based on parent→child relationships
        
        # Root to pelvis: upward
        if parent_name == "root" and child_name == "pelvis":
            return np.array([0.0, 1.0, 0.0])
            
        # Spine chain: upward
        elif "spine" in parent_name and ("spine" in child_name or "neck" in child_name):
            return np.array([0.0, 1.0, 0.0])
            
        # Neck to head: upward
        elif "neck" in parent_name and "head" in child_name:
            return np.array([0.0, 1.0, 0.0])
            
        # Spine to clavicles: horizontal outward
        elif "spine" in parent_name and "clavicle" in child_name:
            if "_l" in child_name:
                return np.array([-1.0, 0.0, 0.0])  # Left clavicle
            else:
                return np.array([1.0, 0.0, 0.0])   # Right clavicle
                
        # Clavicle to upperarm: horizontal outward
        elif "clavicle" in parent_name and "upperarm" in child_name:
            if "_l" in child_name:
                return np.array([-1.0, 0.0, 0.0])  # Left arm
            else:
                return np.array([1.0, 0.0, 0.0])   # Right arm
                
        # Arm chain: horizontal outward (T-pose)
        elif ("upperarm" in parent_name and "lowerarm" in child_name) or \
             ("lowerarm" in parent_name and "hand" in child_name):
            if "_l" in child_name:
                return np.array([-1.0, 0.0, 0.0])  # Left arm extension
            else:
                return np.array([1.0, 0.0, 0.0])   # Right arm extension
                
        # Hand to fingers
        elif "hand" in parent_name and any(finger in child_name for finger in ["thumb", "index", "middle", "ring", "pinky"]):
            if "thumb" in child_name:
                # Thumb points forward and slightly outward
                if "_l" in child_name:
                    return np.array([-0.5, 0.0, 0.866])  # 30° forward from left
                else:
                    return np.array([0.5, 0.0, 0.866])   # 30° forward from right
            else:
                # Other fingers extend along arm direction
                if "_l" in child_name:
                    return np.array([-1.0, 0.0, 0.0])
                else:
                    return np.array([1.0, 0.0, 0.0])
                    
        # Finger joints: continue in same direction
        elif any(finger in parent_name for finger in ["thumb", "index", "middle", "ring", "pinky"]) and \
             any(finger in child_name for finger in ["thumb", "index", "middle", "ring", "pinky"]):
            if "thumb" in parent_name:
                if "_l" in child_name:
                    return np.array([-0.5, 0.0, 0.866])
                else:
                    return np.array([0.5, 0.0, 0.866])
            else:
                if "_l" in child_name:
                    return np.array([-1.0, 0.0, 0.0])
                else:
                    return np.array([1.0, 0.0, 0.0])
                    
        # Pelvis to thighs: downward
        elif "pelvis" in parent_name and "thigh" in child_name:
            return np.array([0.0, -1.0, 0.0])
            
        # Leg chain: downward
        elif ("thigh" in parent_name and "calf" in child_name) or \
             ("calf" in parent_name and "foot" in child_name):
            return np.array([0.0, -1.0, 0.0])
            
        # Foot to ball: forward
        elif "foot" in parent_name and "ball" in child_name:
            return np.array([0.0, 0.0, 1.0])
            
        # Head to facial features
        elif "head" in parent_name and child_name in ["jaw", "eye_l", "eye_r"]:
            if child_name == "jaw":
                return np.array([0.0, -1.0, 0.0])  # Jaw downward
            else:
                return np.array([0.0, 0.0, 1.0])   # Eyes forward
                
        # Spine to breasts: forward
        elif "spine" in parent_name and "breast" in child_name:
            return np.array([0.0, 0.0, 1.0])
            
        # Twist bones: follow parent direction
        elif "twist" in child_name:
            # Get the main bone direction (parent of twist bone)
            main_parent_idx = self.parent_indices[parent_index] if parent_index > 0 else -1
            if main_parent_idx >= 0:
                return self.get_tpose_parent_to_child_direction(main_parent_idx, parent_index)
            else:
                return np.array([0.0, 1.0, 0.0])
                
        # IK bones: upward by default
        elif "ik_" in parent_name or "ik_" in child_name:
            return np.array([0.0, 1.0, 0.0])
            
        # Default: upward
        else:
            return np.array([0.0, 1.0, 0.0])