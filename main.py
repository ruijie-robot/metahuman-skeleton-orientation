#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetaHuman 骨架方向求解器 - 主程序

演示如何使用XML驱动的URDF系统处理MetaHuman骨架动画数据，
将世界坐标转换为局部四元数。

使用方法:
    python main.py
"""

import sys
import os
import numpy as np
from quaternion_solver import QuaternionSolverXML

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

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


def demo_workflow():
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


def main():
    """主函数"""
    try:
        # 运行XML工作流程演示
        success = demo_workflow()
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)