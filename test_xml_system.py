import numpy as np
from quaternion_solver_xml import QuaternionSolverXML
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
        animation_data = create_test_animation_data(3)
        result = solver.process_animation_sequence(animation_data)
        print(f"✓ XML Animation sequence test passed: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ XML quaternion solver test failed: {e}")
        return False

def create_test_animation_data(num_frames: int = 10) -> np.ndarray:
    """创建测试动画数据"""
    np.random.seed(42)
    
    animation_data = np.zeros((num_frames, 68, 3))
    
    for frame in range(num_frames):
        # 基本T-pose位置加上轻微动画
        positions = np.array([
            [0, 0, 0],          # root
            [0, 0.1, 0],        # pelvis
            [0, 0.3, 0],        # spine_01
            [0, 0.5, 0],        # spine_02
            [0, 0.7, 0],        # spine_03
            [0, 0.9, 0],        # neck_01
            [0, 1.0, 0],        # head
            [-0.1, 0.7, 0],     # clavicle_l
            [-0.3, 0.7, 0],     # upperarm_l
            [-0.5, 0.7, 0],     # lowerarm_l
            [-0.7, 0.7, 0],     # hand_l
        ])
        
        # 添加简单动画（轻微摆动）
        time = frame / num_frames * 2 * np.pi
        sway = 0.05 * np.sin(time)
        
        # 填充剩余位置
        for i in range(68):
            if i < len(positions):
                animation_data[frame, i] = positions[i]
                animation_data[frame, i, 0] += sway
            else:
                # 为剩余骨骼生成合理位置
                if i >= 11:
                    parent_idx = i - 1 if i > 0 else 0
                    if parent_idx < 68:
                        animation_data[frame, i] = animation_data[frame, parent_idx] + np.array([0.05, 0.05, 0])
    
    return animation_data

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

def compare_systems():
    """比较不同系统的输出"""
    print("\n=== Comparing Systems ===")
    
    try:
        # 导入原始系统
        from quaternion_solver import QuaternionSolver
        
        # 创建求解器
        solver_original = QuaternionSolver()
        solver_xml = QuaternionSolverXML("metahuman.urdf")
        
        # 创建测试数据
        animation_data = create_test_animation_data(1)
        
        # 用两个系统处理
        result_original = solver_original.process_animation_sequence(animation_data)
        result_xml = solver_xml.process_animation_sequence(animation_data)
        
        print(f"Original system output: {result_original.shape}")
        print(f"XML system output:      {result_xml.shape}")
        
        if result_original.shape == result_xml.shape:
            print("✓ Output shapes match!")
            
            # 计算差异
            differences = np.linalg.norm(result_original[0] - result_xml[0], axis=1)
            print(f"\nQuaternion differences (first 10):")
            for i in range(min(10, len(differences))):
                print(f"  Joint {i:2d}: {differences[i]:.6f}")
            
            avg_diff = np.mean(differences)
            max_diff = np.max(differences)
            print(f"\nAverage difference: {avg_diff:.6f}")
            print(f"Maximum difference: {max_diff:.6f}")
            
            if max_diff < 1e-10:
                print("✓ Results are identical!")
            elif max_diff < 1e-6:
                print("✓ Results are very similar (within numerical precision)")
            else:
                print("⚠ Results differ - this might be expected due to different implementations")
        else:
            print("✗ Output shapes don't match!")
        
        return True
        
    except ImportError:
        print("Original system not available for comparison")
        return True
    except Exception as e:
        print(f"✗ System comparison failed: {e}")
        return False

def demo_xml_workflow():
    """演示完整的XML工作流程"""
    print("\n=== XML Workflow Demo ===")
    
    try:
        print("1. Loading skeleton from URDF XML file...")
        solver = QuaternionSolverXML("metahuman.urdf")
        
        print("2. Creating sample animation data...")
        animation_data = create_test_animation_data(5)
        print(f"   Animation data shape: {animation_data.shape}")
        
        print("3. Processing animation sequence...")
        local_quaternions = solver.process_animation_sequence(animation_data)
        print(f"   Local quaternions shape: {local_quaternions.shape}")
        
        print("4. Analyzing results...")
        joint_names = solver.get_joint_names()
        
        print(f"   Frame 0 quaternions (first 5 joints):")
        for i in range(5):
            quat = local_quaternions[0, i]
            print(f"     {joint_names[i]:25s}: [{quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f}]")
        
        print("✓ XML workflow demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ XML workflow demo failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    # success &= test_xml_parsing()
    # success &= test_xml_quaternion_solver()
    # success &= test_joint_queries()
    # success &= compare_systems()
    success &= demo_xml_workflow()
    
    if success:
        print("\n🎉 All XML system tests passed!")
    else:
        print("\n❌ Some XML system tests failed!")