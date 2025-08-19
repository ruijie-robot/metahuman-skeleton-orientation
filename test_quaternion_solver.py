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
        print(f"  Total connections: {len(local_quats)} (expected 67)")
        
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
    print(f"Frame 0, Connection 0 (root→pelvis): {local_quaternions[0, 0]}")
    print(f"Frame 0, Connection 1 (pelvis→spine_01): {local_quaternions[0, 1]}")
    
    return local_quaternions

def test_quaternion_methods():
    """Compare standard and optimized quaternion calculation methods"""
    print("\n--- Comparing Quaternion Calculation Methods ---")
    
    solver = QuaternionSolver()
    
    # Test cases: different rotation angles
    test_cases = [
        ([1, 0, 0], [0, 1, 0], "90° rotation (X to Y)"),
        ([1, 0, 0], [0.866, 0.5, 0], "30° rotation"),
        ([1, 0, 0], [0.707, 0.707, 0], "45° rotation"),
        ([1, 0, 0], [0, 0, 1], "90° rotation (X to Z)"),
        ([1, 0, 0], [-1, 0, 0], "180° rotation"),
        ([1, 0, 0], [1, 0, 0], "0° rotation (no change)"),
    ]
    
    print(f"{'Test Case':<25} {'Standard Method':<25} {'Optimized Method':<25} {'Difference':<15}")
    print("-" * 95)
    
    for vec_from, vec_to, description in test_cases:
        vec_from = np.array(vec_from, dtype=float)
        vec_to = np.array(vec_to, dtype=float)
        
        # Calculate using both methods
        q_standard = solver.quaternion_from_vectors_standard(vec_from, vec_to)
        q_optimized = solver.quaternion_from_vectors_optimized(vec_from, vec_to)
        
        # Calculate difference
        diff = np.linalg.norm(q_standard - q_optimized)
        
        print(f"{description:<25} {str(np.round(q_standard, 3)):<25} {str(np.round(q_optimized, 3)):<25} {diff:<15.6f}")
    
    print("\nNote: Small differences are due to floating-point precision.")

def test_tpose_directions():
    """Test T-pose parent-child bone directions"""
    print("\n--- Testing T-pose Parent-Child Directions ---")
    
    skeleton = MetahumanSkeleton()
    
    # Test key parent→child connections
    test_connections = [
        (0, 1, "root → pelvis"),
        (1, 2, "pelvis → spine_01"),
        (4, 7, "spine_03 → clavicle_l"),
        (4, 26, "spine_03 → clavicle_r"),
        (7, 8, "clavicle_l → upperarm_l"),
        (26, 27, "clavicle_r → upperarm_r"),
        (8, 9, "upperarm_l → lowerarm_l"),
        (27, 28, "upperarm_r → lowerarm_r"),
        (1, 45, "pelvis → thigh_l"),
        (1, 49, "pelvis → thigh_r"),
        (45, 46, "thigh_l → calf_l"),
        (5, 6, "neck_01 → head"),
    ]
    
    print(f"{'Connection':<30} {'T-pose Direction':<20} {'Description'}")
    print("-" * 75)
    
    for parent_idx, child_idx, description in test_connections:
        direction = skeleton.get_tpose_parent_to_child_direction(parent_idx, child_idx)
        direction_str = f"[{direction[0]:5.1f}, {direction[1]:5.1f}, {direction[2]:5.1f}]"
        
        # Describe the direction
        desc = ""
        if np.allclose(direction, [0, 1, 0]):
            desc = "Upward"
        elif np.allclose(direction, [0, -1, 0]):
            desc = "Downward"
        elif np.allclose(direction, [-1, 0, 0]):
            desc = "Left"
        elif np.allclose(direction, [1, 0, 0]):
            desc = "Right"
        elif np.allclose(direction, [0, 0, 1]):
            desc = "Forward"
        else:
            desc = "Custom"
            
        print(f"{description:<30} {direction_str:<20} {desc}")
    
    print(f"\nTotal bone count: {len(skeleton.bone_names)}")
    
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
    print(f"Connection 7 (clavicle_l→upperarm_l, T-pose): {result_tpose[0, 7]}")
    
    print("\nArm-down quaternions:")
    result_down = solver.process_animation_sequence(animation_data_down)
    print(f"Connection 7 (clavicle_l→upperarm_l, arm down): {result_down[0, 7]}")

if __name__ == "__main__":
    test_quaternion_solver()
    demo_usage()
    test_quaternion_methods()
    test_tpose_directions()