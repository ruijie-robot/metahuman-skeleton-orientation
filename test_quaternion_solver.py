import numpy as np
from quaternion_solver import QuaternionSolver
from metahuman_skeleton import MetahumanSkeleton

def create_test_animation_data(num_frames: int = 10) -> np.ndarray:
    """Create sample animation data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Create a simple T-pose with some animation
    animation_data = np.zeros((num_frames, 68, 3))
    
    for frame in range(num_frames):
        # Basic T-pose positions with slight animation
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
        
        # Add simple animation (slight swaying)
        time = frame / num_frames * 2 * np.pi
        sway = 0.05 * np.sin(time)
        
        # Fill remaining positions with interpolated/extrapolated values
        for i in range(68):
            if i < len(positions):
                animation_data[frame, i] = positions[i]
                animation_data[frame, i, 0] += sway  # Add swaying motion
            else:
                # Generate reasonable positions for remaining bones
                if i >= 11:  # fingers and other detailed bones
                    parent_idx = i - 1 if i > 0 else 0
                    if parent_idx < 68:
                        # Position slightly offset from parent
                        animation_data[frame, i] = animation_data[frame, parent_idx] + np.array([0.05, 0.05, 0])
    
    return animation_data

def test_quaternion_solver():
    """Test the quaternion solver implementation"""
    print("Testing Quaternion Solver...")
    
    # Create test data
    solver = QuaternionSolver()
    skeleton = MetahumanSkeleton()
    
    print(f"Skeleton has {len(skeleton.bone_names)} bones")
    print(f"Bone names: {skeleton.bone_names[:10]}...")  # Show first 10
    
    # Test single frame
    test_positions = np.random.rand(68, 3)
    test_positions[:, 1] = np.abs(test_positions[:, 1])  # Ensure positive Y (up)
    
    try:
        local_quats = solver.world_to_local_quaternions(test_positions)
        print(f"✓ Single frame test passed: {local_quats.shape}")
        print(f"  Sample quaternion: {local_quats[0]}")
        
        # Verify quaternions are normalized
        norms = np.linalg.norm(local_quats, axis=1)
        print(f"  Quaternion norms (should be ~1.0): min={norms.min():.3f}, max={norms.max():.3f}")
        
    except Exception as e:
        print(f"✗ Single frame test failed: {e}")
        return False
    
    # Test animation sequence
    try:
        animation_data = create_test_animation_data(5)
        print(f"✓ Created test animation data: {animation_data.shape}")
        
        result = solver.process_animation_sequence(animation_data)
        print(f"✓ Animation sequence test passed: {result.shape}")
        
        # Check for reasonable quaternion values
        print(f"  Quaternion value range: [{result.min():.3f}, {result.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Animation sequence test failed: {e}")
        return False
    
    print("All tests passed!")
    return True

def demo_usage():
    """Demonstrate usage of the quaternion solver"""
    print("\n--- Usage Demo ---")
    
    solver = QuaternionSolver()
    
    # Create sample data (5 frames, 68 bones, 3D positions)
    animation_data = create_test_animation_data(5)
    
    print(f"Input: Animation data shape {animation_data.shape}")
    print(f"Frame 0, Bone 0 (root) position: {animation_data[0, 0]}")
    print(f"Frame 0, Bone 1 (pelvis) position: {animation_data[0, 1]}")
    
    # Convert to local quaternions
    local_quaternions = solver.process_animation_sequence(animation_data)
    
    print(f"Output: Local quaternions shape {local_quaternions.shape}")
    print(f"Frame 0, Bone 0 (root) quaternion: {local_quaternions[0, 0]}")
    print(f"Frame 0, Bone 1 (pelvis) quaternion: {local_quaternions[0, 1]}")
    
    return local_quaternions

def test_tpose_directions():
    """Test T-pose bone directions"""
    print("\n--- Testing T-pose Bone Directions ---")
    
    skeleton = MetahumanSkeleton()
    
    # Test a few key bones
    test_bones = [
        (1, "pelvis"),      # Should be Y-up
        (8, "upperarm_l"),  # Should be X-left  
        (27, "upperarm_r"), # Should be X-right
        (45, "thigh_l"),    # Should be Y-down
        (5, "neck_01"),     # Should be Y-up
    ]
    
    for bone_idx, bone_name in test_bones:
        direction = skeleton.get_tpose_bone_direction(bone_idx)
        print(f"Bone {bone_idx:2d} ({bone_name:12s}): {direction}")
    
    print("\nTest animation with different bone orientations...")
    
    # Create test data with specific bone orientations
    animation_data = np.zeros((1, 68, 3))
    
    # Root at origin
    animation_data[0, 0] = [0, 0, 0]
    # Pelvis slightly above
    animation_data[0, 1] = [0, 0.1, 0]
    # Left upperarm: T-pose position (horizontal left)
    animation_data[0, 7] = [-0.1, 0.7, 0]  # clavicle_l
    animation_data[0, 8] = [-0.3, 0.7, 0]  # upperarm_l
    animation_data[0, 9] = [-0.5, 0.7, 0]  # lowerarm_l
    
    # Test different arm position (arm down)
    animation_data_down = animation_data.copy()
    animation_data_down[0, 9] = [-0.3, 0.5, 0]  # lowerarm_l moved down
    
    solver = QuaternionSolver()
    
    print("\nT-pose quaternions:")
    result_tpose = solver.process_animation_sequence(animation_data)
    print(f"Upperarm_l quaternion (T-pose): {result_tpose[0, 8]}")
    
    print("\nArm-down quaternions:")
    result_down = solver.process_animation_sequence(animation_data_down)
    print(f"Upperarm_l quaternion (arm down): {result_down[0, 8]}")

if __name__ == "__main__":
    test_quaternion_solver()
    demo_usage()
    test_tpose_directions()