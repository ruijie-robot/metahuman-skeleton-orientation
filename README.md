# Metahuman Skeleton Orientation Solver

This project solves the problem of converting world coordinates of metahuman skeleton bones to local quaternions for animation sequences.

## Problem Statement

Given:
1. World coordinates (x, y, z) for 67 main skeletal points of metahuman skeleton in each animation frame
2. Parent-child bone hierarchy relationships

Solve for:
- Local quaternions for all 68 skeletal points (67 main bones + root) in each frame

## Files

- `metahuman_skeleton.py`: Defines the metahuman skeleton structure with 68 bones and their hierarchy
- `quaternion_solver.py`: Core algorithm for converting world coordinates to local quaternions
- `test_quaternion_solver.py`: Test script and usage examples

## Usage

```python
from quaternion_solver import QuaternionSolver
import numpy as np

# Initialize solver
solver = QuaternionSolver()

# Your animation data: shape (num_frames, 68, 3)
animation_data = np.array(...)  # Load your world coordinate data

# Convert to local quaternions: shape (num_frames, 68, 4)
local_quaternions = solver.process_animation_sequence(animation_data)

# local_quaternions[frame_idx, bone_idx] = [w, x, y, z] quaternion
```

## Algorithm Overview

1. **Bone Hierarchy**: Uses predefined metahuman skeleton with 68 bones including root
2. **Orientation Calculation**: Computes bone orientation from bone-to-child direction vectors
3. **Local Space Conversion**: Transforms world quaternions to local space relative to parent bones
4. **Hierarchical Processing**: Processes bones in hierarchical order to maintain parent-child relationships

## Testing

Run the test script:
```bash
python test_quaternion_solver.py
```

The test verifies:
- Correct quaternion normalization
- Proper handling of animation sequences
- Reasonable output values