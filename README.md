# MetaHuman 骨架方向求解器

Author: Ruijie

本项目解决将 MetaHuman 骨架的世界坐标转换为动画序列中局部四元数的问题。

## 问题描述

**输入**：
1. 动画序列中每一帧人体68个骨骼点的世界坐标系坐标 (x, y, z)
2. 人体68个骨骼点的父子关系（通过URDF文件定义）

**输出**：
- 动画序列中每一帧人体68个骨骼点的局部四元数（包括root骨骼 + 67个关节连接）

## 系统架构
### `main.py`: 主程序入口
### `metahuman.urdf`: 标准URDF XML文件，定义68个骨骼和67个关节
### `urdf_parser.py`: URDF XML解析器
### `quaternion_solver.py`: 基于XML的四元数求解器
### `test.py`: XML系统完整测试

## 快速开始
```
python main.py
```



## 骨架结构

Author: Ruijie

### MetaHuman 骨架包含：
- **68个骨骼** (Links): 从root到手指尖端的完整人体骨架
- **67个关节** (Joints): 连接骨骼的旋转关节

### 主要骨骼链：
```
root
└── pelvis
    ├── spine_01 → spine_02 → spine_03 → neck_01 → head
    ├── clavicle_l → upperarm_l → lowerarm_l → hand_l → fingers_l
    ├── clavicle_r → upperarm_r → lowerarm_r → hand_r → fingers_r  
    ├── thigh_l → calf_l → foot_l → ball_l
    └── thigh_r → calf_r → foot_r → ball_r
```

## 算法原理

Author: Ruijie

### 1. T-pose 参考系统
- 使用Unity坐标系: X-右, Y-上, Z-前
- 每个关节定义T-pose中的标准方向
- 例如：左臂在T-pose中水平向左延伸

### 2. 四元数计算
- 根据T-pose, 每一个骨骼点都有自己初始的位姿（世界坐标系），而每一个父子关系是一个局部坐标系，局部坐标系以父骨骼点为局部坐标系的原点，可以求得子骨节点相对于原点的向量v0；

- 在动画序列里面每一次运动，每个骨骼点都会有新的世界坐标位置信息 (x, y, z), 从而得到新的向量vt；

- 而vt相对于初始位置v0的旋转，就是这个子骨节点在局部坐标系下的局部四元数;
使用标准四元数公式：
```
q = [cos(θ/2), u_x * sin(θ/2), u_y * sin(θ/2), u_z * sin(θ/2)]
```
其中 `axis = (u_x, u_y, u_z) = normalize(初始方向 × 当前方向)`

- 同时，除了67个joints之外，还有一个root骨骼点，root没有父骨骼点，可以认为root是相对于世界坐标系下来说的，所以root的局部四元数等于世界坐标系下的四元数。


## URDF文件格式

Author: Ruijie

### 为什么使用URDF格式来定义骨骼层？
`metahuman.urdf` 使用标准URDF XML格式, 
URDF（Unified Robot Description Format）是机器人领域的标准骨架描述格式，广泛应用于各种机器人仿真引擎（如ROS、Gazebo、Isaac等）。使用URDF定义骨骼结构的最大优势在于：

- **与机器人仿真引擎无缝对接**：URDF是主流机器人仿真平台的通用输入格式，直接支持骨骼、关节、运动学等信息的解析和加载。这样可以方便地将动画骨架与机器人仿真环境集成，实现动作复现、物理仿真和可视化。
- **标准化与互操作性**：采用URDF可以让骨架结构与机器人领域的工具链（如运动学求解、动力学仿真、传感器模拟等）高度兼容，便于后续扩展和跨平台应用。
- **结构清晰、易于维护**：URDF的层次化XML结构使得每个link和joint都清晰可见，便于编辑、调试和自动化处理。
- **支持自定义属性**：可以在URDF中扩展自定义字段（如T-pose方向），满足动画和仿真结合的需求。

### URDF里面的link和joint
在URDF文件中，`link` 和 `joint` 分别对应于MetaHuman骨骼系统中的“骨骼节点”和“关节连接”：

- **link（骨骼/链节）**  
  每一个 `link` 标签代表一个骨骼点（如 root、pelvis、upperarm_l、hand_r 等），它们是骨架结构中的节点。对于MetaHuman来说，68个link就对应了68个人体关键骨骼点，包括躯干、四肢、手指、头部等。

- **joint（关节/连接）**  
  每一个 `joint` 标签描述了两个骨骼点之间的连接关系（即父子关系），并定义了它们之间的旋转轴、初始方向（T-pose下的朝向）等属性。67个joint就对应了67个人体骨骼之间的旋转关节，比如“upperarm_l_to_lowerarm_l”表示左上臂和左下臂之间的肘关节。

**对应关系举例：**
- `link name="upperarm_l"` 表示左上臂的骨骼点。
- `joint name="clavicle_l_to_upperarm_l"` 表示左锁骨到左上臂的关节连接，`parent link="clavicle_l"`，`child link="upperarm_l"`，并指定了T-pose下的方向。

**结构关系：**
- link负责定义“点”，joint负责定义“点与点之间的连接和旋转关系”。
- 通过joint的parent/child属性，URDF文件描述了完整的人体骨架树状结构，和MetaHuman骨骼系统一一对应。

**自定义属性：**
- 在每个joint下的 `<tpose_direction xyz="..."/>` 字段，专门用于动画/仿真场景，明确T-pose下该关节的标准朝向，便于后续四元数计算。

**总结：**
- URDF中的link和joint严格一一映射MetaHuman骨骼点和关节关系，保证了骨架结构的完整性和可解析性，方便动画数据与仿真系统的对接和处理。




## 常见问题

Author: Ruijie

### Q: 为什么返回68个四元数而不是67个？
A: 虽然只有67个关节连接，但我们额外包含了root骨骼的四元数，用于表示整个人体在世界空间中的朝向。

