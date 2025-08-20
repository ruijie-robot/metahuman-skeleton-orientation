import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class URDFLink:
    """URDF Link (骨骼/链节)"""
    name: str
    index: int

@dataclass 
class URDFJoint:
    """URDF Joint (关节/连接)"""
    name: str
    joint_type: str  # "revolute", "prismatic", "fixed", etc.
    parent_link: str
    child_link: str
    parent_index: int
    child_index: int
    axis: np.ndarray  # 旋转轴 [x, y, z]
    tpose_direction: np.ndarray  # T-pose中parent→child的方向
    origin: Optional[np.ndarray] = None  # 位置偏移 [x, y, z]

class URDFParser:
    """URDF XML文件解析器"""
    
    def __init__(self, urdf_file_path: str):
        self.urdf_file_path = urdf_file_path
        self.links: List[URDFLink] = []
        self.joints: List[URDFJoint] = []
        self.link_name_to_index: Dict[str, int] = {}
        self.joint_name_to_index: Dict[str, int] = {}
        
        self.parse_urdf()
    
    def parse_urdf(self):
        """解析URDF XML文件"""
        try:
            tree = ET.parse(self.urdf_file_path)
            root = tree.getroot()
            
            if root.tag != 'robot':
                raise ValueError("URDF file must have 'robot' as root element")
            
            # 解析所有links
            self._parse_links(root)
            
            # 解析所有joints
            self._parse_joints(root)
            
            # 创建索引映射
            self._create_indices()
            
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse URDF XML: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"URDF file not found: {self.urdf_file_path}")
    
    def _parse_links(self, root: ET.Element):
        """解析所有link元素"""
        links = root.findall('link')
        
        for i, link_elem in enumerate(links):
            name = link_elem.get('name')
            if not name:
                raise ValueError(f"Link at index {i} missing 'name' attribute")
            
            urdf_link = URDFLink(name=name, index=i)
            self.links.append(urdf_link)
    
    def _parse_joints(self, root: ET.Element):
        """解析所有joint元素"""
        joints = root.findall('joint')
        
        for joint_elem in joints:
            name = joint_elem.get('name')
            joint_type = joint_elem.get('type')
            
            if not name or not joint_type:
                raise ValueError(f"Joint missing 'name' or 'type' attribute")
            
            # 解析parent和child
            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            
            if parent_elem is None or child_elem is None:
                raise ValueError(f"Joint '{name}' missing parent or child")
            
            parent_link = parent_elem.get('link')
            child_link = child_elem.get('link')
            
            if not parent_link or not child_link:
                raise ValueError(f"Joint '{name}' missing parent or child link name")
            
            # 解析axis (旋转轴)
            axis_elem = joint_elem.find('axis')
            axis = np.array([0.0, 0.0, 1.0])  # 默认Z轴
            if axis_elem is not None:
                axis_str = axis_elem.get('xyz', '0 0 1')
                axis = np.array([float(x) for x in axis_str.split()])
            
            # 解析origin (位置偏移)
            origin_elem = joint_elem.find('origin')
            origin = None
            if origin_elem is not None:
                origin_str = origin_elem.get('xyz', '0 0 0')
                origin = np.array([float(x) for x in origin_str.split()])
            
            # 解析自定义的tpose_direction
            tpose_elem = joint_elem.find('tpose_direction')
            tpose_direction = np.array([0.0, 1.0, 0.0])  # 默认向上
            if tpose_elem is not None:
                tpose_str = tpose_elem.get('xyz', '0 1 0')
                tpose_direction = np.array([float(x) for x in tpose_str.split()])
            
            urdf_joint = URDFJoint(
                name=name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                parent_index=-1,  # 将在_create_indices中设置
                child_index=-1,   # 将在_create_indices中设置
                axis=axis,
                tpose_direction=tpose_direction,
                origin=origin
            )
            
            self.joints.append(urdf_joint)
    
    def _create_indices(self):
        """创建名称到索引的映射"""
        # 创建link名称到索引的映射
        self.link_name_to_index = {link.name: link.index for link in self.links}
        
        # 为joints设置parent和child的索引
        for joint in self.joints:
            if joint.parent_link not in self.link_name_to_index:
                raise ValueError(f"Parent link '{joint.parent_link}' not found for joint '{joint.name}'")
            if joint.child_link not in self.link_name_to_index:
                raise ValueError(f"Child link '{joint.child_link}' not found for joint '{joint.name}'")
            
            joint.parent_index = self.link_name_to_index[joint.parent_link]
            joint.child_index = self.link_name_to_index[joint.child_link]
        
        # 创建joint名称到索引的映射
        self.joint_name_to_index = {joint.name: i for i, joint in enumerate(self.joints)}
    
    def get_link_by_name(self, name: str) -> Optional[URDFLink]:
        """通过名称获取link"""
        for link in self.links:
            if link.name == name:
                return link
        return None
    
    def get_joint_by_name(self, name: str) -> Optional[URDFJoint]:
        """通过名称获取joint"""
        for joint in self.joints:
            if joint.name == name:
                return joint
        return None
    
    def get_joint_by_child_index(self, child_index: int) -> Optional[URDFJoint]:
        """通过子link索引获取joint"""
        for joint in self.joints:
            if joint.child_index == child_index:
                return joint
        return None
    
    def get_children_joints(self, parent_index: int) -> List[URDFJoint]:
        """获取指定parent的所有子joints"""
        children = []
        for joint in self.joints:
            if joint.parent_index == parent_index:
                children.append(joint)
        return children
    
    def print_summary(self):
        """打印URDF结构摘要"""
        print(f"URDF Parser Summary:")
        print(f"  File: {self.urdf_file_path}")
        print(f"  Links: {len(self.links)}")
        print(f"  Joints: {len(self.joints)}")
        
        print(f"\nFirst 10 links:")
        for i, link in enumerate(self.links[:10]):
            print(f"  {i:2d}: {link.name}")
        
        if len(self.links) > 10:
            print(f"  ... and {len(self.links) - 10} more links")
        
        print(f"\nFirst 10 joints:")
        for i, joint in enumerate(self.joints[:10]):
            direction_str = f"[{joint.tpose_direction[0]:4.1f}, {joint.tpose_direction[1]:4.1f}, {joint.tpose_direction[2]:4.1f}]"
            print(f"  {i:2d}: {joint.parent_link:12s} → {joint.child_link:15s} | {direction_str}")
        
        if len(self.joints) > 10:
            print(f"  ... and {len(self.joints) - 10} more joints")
    
    def validate_structure(self) -> bool:
        """验证URDF结构的完整性"""
        try:
            # 检查是否有root link (没有parent的link)
            root_links = []
            for link in self.links:
                has_parent = False
                for joint in self.joints:
                    if joint.child_index == link.index:
                        has_parent = True
                        break
                if not has_parent:
                    root_links.append(link)
            
            if len(root_links) != 1:
                print(f"Warning: Expected 1 root link, found {len(root_links)}")
                for root in root_links:
                    print(f"  Root: {root.name}")
            
            # 检查所有joints的parent和child是否有效
            for joint in self.joints:
                if joint.parent_index < 0 or joint.parent_index >= len(self.links):
                    print(f"Error: Joint '{joint.name}' has invalid parent index {joint.parent_index}")
                    return False
                if joint.child_index < 0 or joint.child_index >= len(self.links):
                    print(f"Error: Joint '{joint.name}' has invalid child index {joint.child_index}")
                    return False
            
            print("✓ URDF structure validation passed")
            return True
            
        except Exception as e:
            print(f"✗ URDF structure validation failed: {e}")
            return False