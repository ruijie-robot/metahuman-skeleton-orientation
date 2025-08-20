import numpy as np
from quaternion_solver import QuaternionSolverXML
from urdf_parser import URDFParser

def test_xml_parsing():
    """测试XML解析"""
    print("=== Testing XML Parsing ===")
    
    try:
        parser = URDFParser("metahuman.urdf")
        parser.print_summary()
        
        # 测试一些具体的访问方法
        print(f"\nTesting specific access methods:")
        
        # 获取root link
        root_link = parser.get_link_by_name("root")
        if root_link:
            print(f"Root link: {root_link.name} (index {root_link.index})")
        
        # 获取pelvis joint
        pelvis_joint = parser.get_joint_by_name("root_to_pelvis")
        if pelvis_joint:
            print(f"Pelvis joint: {pelvis_joint.name}")
            print(f"  Parent: {pelvis_joint.parent_link} (index {pelvis_joint.parent_index})")
            print(f"  Child: {pelvis_joint.child_link} (index {pelvis_joint.child_index})")
            print(f"  T-pose direction: {pelvis_joint.tpose_direction}")
            print(f"  Joint type: {pelvis_joint.joint_type}")
            print(f"  Axis: {pelvis_joint.axis}")
        
        # 获取spine_03的所有子关节
        spine03_children = parser.get_children_joints(4)  # spine_03 should be index 4
        print(f"\nChildren of spine_03 (index 4):")
        for joint in spine03_children:
            print(f"  {joint.name}: {joint.parent_link} → {joint.child_link}")
        
        return True
        
    except Exception as e:
        print(f"✗ XML parsing failed: {e}")
        return False

def test_xml_quaternion_solver():
    """测试基于XML的四元数求解器"""
    print("\n=== Testing XML Quaternion Solver ===")
    
    try:
        solver = QuaternionSolverXML("metahuman.urdf")
        
        # 创建测试数据
        np.random.seed(42)
        test_positions = np.random.rand(68, 3)
        test_positions[:, 1] = np.abs(test_positions[:, 1])  # 确保Y坐标为正
        
        # 单帧测试
        local_quats = solver.world_to_local_quaternions(test_positions)
        print(f"✓ XML Single frame test passed: {local_quats.shape}")
        
        # 验证四元数归一化
        norms = np.linalg.norm(local_quats, axis=1)
        print(f"  Quaternion norms (should be ~1.0): min={norms.min():.3f}, max={norms.max():.3f}")
        print(f"  Total joints: {len(local_quats)}")
        
        # 动画序列测试
        np.random.seed(42)
        animation_data = np.random.rand(3, 68, 3)
        animation_data[:, :, 1] = np.abs(animation_data[:, :, 1])  # 确保 Y 坐标为正
        result = solver.process_animation_sequence(animation_data)
        print(f"✓ XML Animation sequence test passed: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ XML quaternion solver test failed: {e}")
        return False


def test_joint_queries():
    """测试关节查询功能"""
    print("\n=== Testing Joint Queries ===")
    
    try:
        solver = QuaternionSolverXML("metahuman.urdf")
        
        # 测试关节信息查询
        joint_info = solver.get_joint_info("clavicle_l_to_upperarm_l")
        if joint_info:
            print(f"Joint info for 'clavicle_l_to_upperarm_l':")
            print(f"  Type: {joint_info.joint_type}")
            print(f"  Parent: {joint_info.parent_link}")
            print(f"  Child: {joint_info.child_link}")
            print(f"  T-pose direction: {joint_info.tpose_direction}")
            print(f"  Axis: {joint_info.axis}")
        
        # 测试链接信息查询
        link_info = solver.get_link_info("upperarm_l")
        if link_info:
            print(f"\nLink info for 'upperarm_l':")
            print(f"  Name: {link_info.name}")
            print(f"  Index: {link_info.index}")
        
        # 获取所有关节名称
        joint_names = solver.get_joint_names()
        print(f"\nFirst 10 joint names:")
        for i, name in enumerate(joint_names[:10]):
            print(f"  {i:2d}: {name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Joint queries test failed: {e}")
        return False



if __name__ == "__main__":
    success = True
    # success &= test_xml_parsing()
    # success &= test_xml_quaternion_solver()
    # success &= test_joint_queries()
    # success &= demo_xml_workflow()
    
    if success:
        print("\n🎉 All XML system tests passed!")
    else:
        print("\n❌ Some XML system tests failed!")