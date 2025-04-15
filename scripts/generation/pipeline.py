import sys
sys.path.append('/cpfs01/user/yenaisheng/SceneGen')
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan_extent.planning.scenario_builder.prepared_scenario import (
    NpEgoState,
    PreparedScenario,
)
from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_vector_feature_builder_v2 import (
    HorizonVectorFeatureBuilderV2,
    HorizonVectorV2,
)
from nuplan_extent.planning.training.preprocessing.features.scene_tensor import (SceneTensor,
    SCENE_TENSOR_FEATURES,N_SCENE_TENSOR_FEATURES,  
    FEATURE_MEANS,FEATURE_STD,
    VEH_PARAMS,EGO_WIDTH,EGO_LENGTH,CLASSES,
    unnormalize_roadgraph,decode_scene_tensor,encode_agent,encode_ego) 
from typing import List, Dict
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.cuda.amp import autocast
from collections import Counter

@dataclass
class SceneTensorWrapper:
    scene_tensor: SceneTensor
    trans: np.ndarray
    center: np.ndarray
    offset: np.ndarray
    idx2token: List[str]
    log_name: str
    scenario_token: str
    map_name: str
    lidar_pc: List[str]
    agent_type: List[str]

# exp: visualize_scene_tensor(scene_tensor)
def visualize_scene_tensor(scene_tensor, output_gif="scene.gif", fps=2):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import imageio
    """
        可视化场景中车辆的边界框，并生成 GIF 动画。
        
        参数：
            scene_tensor: SceneTensor 对象，其中包含：
                - tensor: 形状 (NA, 21, 8)，每个车辆在 21 个时间戳下的状态，
                        各维度依次为 x, y, cos(yaw), sin(yaw), vx, vy, length, width
                - validity: 形状 (NA, 21)，表示对应车辆在每个时刻是否有效
                - road_graph 和 road_graph_validity 可根据需要扩展（此处不作处理）
            output_gif: 输出的 GIF 文件名，默认为 "scene.gif"
            fps: GIF 动画的帧率，默认为 2 帧每秒
        """
    
    # 获取车辆状态和有效性信息
    # 假设 decode_scene_tensor 返回的对象具有 tensor 和 validity 两个属性
    tensor = decode_scene_tensor(scene_tensor.tensor)      # shape: (NA, 21, 8)
    validity = scene_tensor.validity  # shape: (NA, 21)
    
    num_vehicles, num_timesteps, _ = tensor.shape

    # 为了保证所有帧的坐标轴一致，这里计算所有时刻车辆中心点的全局范围
    all_x = tensor[:, :, 0]
    all_y = tensor[:, :, 1]
    margin = 10  # 预留边界
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin

    frames = []  # 用来存储每一帧图片

    for t in range(num_timesteps):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"Time step {t}")
        
        # 遍历每一辆车
        for i in range(num_vehicles):
            # 如果当前时刻该车辆无效，则跳过
            if validity[i, t] < 0.5:
                continue

            # 提取车辆状态信息
            x = tensor[i, t, 0]
            y = tensor[i, t, 1]
            cos_yaw = tensor[i, t, 2]
            sin_yaw = tensor[i, t, 3]
            # vx 和 vy 此处不用于绘制
            length = tensor[i, t, 6]
            width = tensor[i, t, 7]
            # 计算偏航角（车辆朝向），以弧度表示
            yaw = np.arctan2(sin_yaw, cos_yaw)
            yaw_deg = np.degrees(yaw)

            # 计算 bbox 的绘制参数：
            # matplotlib.patches.Rectangle 要求提供左下角坐标，
            # 而车辆状态给出的 (x, y) 是中心点，所以需要转换：
            half_length = length / 2
            half_width = width / 2
            # 根据旋转矩形公式：
            # 左下角坐标 = center - [half_length*cos(yaw) - half_width*sin(yaw), half_length*sin(yaw) + half_width*cos(yaw)]
            bottom_left_x = x - (half_length * np.cos(yaw) - half_width * np.sin(yaw))
            bottom_left_y = y - (half_length * np.sin(yaw) + half_width * np.cos(yaw))
            
            # 创建表示车辆 bbox 的矩形补丁（不填充，仅显示边框）
            rect = patches.Rectangle(
                (bottom_left_x, bottom_left_y), 
                length, width, 
                angle=yaw_deg, 
                edgecolor='red', 
                facecolor='none', 
                lw=2
            )
            ax.add_patch(rect)
            # 绘制车辆中心点，便于观察车辆位置
            ax.plot(x, y, 'bo')
        
        # 将当前绘图转换为图像帧
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    
    # 使用 imageio 生成 GIF 动画
    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"GIF 动画已保存为 {output_gif}")

# exp: visualize_scene_tensor_global(scene_tensor, trans, center, offset)
def visualize_scene_tensor_global(scene_tensor, local_to_global, center, global_offset, output_gif="scene_global.gif", fps=2):
    """
    将已解码的 scene_tensor 从 local 坐标系转换到 global 坐标系后进行可视化，并生成 GIF 动画。

    参数：
        scene_tensor: 已解码的 SceneTensor 对象，包含：
            - tensor: numpy 数组，形状 (NA, 21, 8)，每个车辆在 21 个时间戳下的状态，
                      各维度依次为 x, y, cos(yaw), sin(yaw), vx, vy, length, width
            - validity: numpy 数组，形状 (NA, 21)，表示车辆在每个时刻是否有效
        local_to_global: 3×3 的 numpy 数组，local 到 global 坐标系的齐次变换矩阵
        center: 全局坐标系下的中心参考对象，须包含属性 heading（以弧度表示），用于调整车辆朝向
        global_offset: numpy 数组，形状 (2,)，表示在 global 坐标系下 x, y 的偏移量
        output_gif: 输出的 GIF 文件名，默认为 "scene_global.gif"
        fps: GIF 的帧率，默认为 2 帧每秒
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import imageio

    # 获取原始 tensor 和有效性信息（tensor 中的数据均在 local 坐标系下）
    tensor = decode_scene_tensor(scene_tensor.tensor)      # shape: (NA, 21, 8)
    validity = scene_tensor.validity  # shape: (NA, 21)
    num_vehicles, num_timesteps, _ = tensor.shape

    # --- 坐标转换 ---
    # 1. 位置转换：对 tensor 中的 x,y (shape: (NA, T, 2)) 使用齐次坐标变换
    xy_local = tensor[:, :, 0:2]  # local 坐标 (x, y)
    ones = np.ones((num_vehicles, num_timesteps, 1))
    xy_local_h = np.concatenate([xy_local, ones], axis=-1)  # (NA, T, 3)
    # 使用 np.einsum 对每个点进行 local_to_global 变换
    xy_global_h = np.einsum('ij,ntj->nti', local_to_global, xy_local_h)
    xy_global = xy_global_h[:, :, :2]  # 提取转换后的 (x, y)
    # 加上全局偏移量
    xy_global = xy_global + global_offset  # broadcast 偏移 (2,)

    # 2. 朝向转换：local 的 yaw 需要加上 center.heading
    local_yaw = np.arctan2(tensor[:, :, 3], tensor[:, :, 2])  # shape: (NA, T)
    global_yaw = local_yaw + center.heading
    global_cos = np.cos(global_yaw)
    global_sin = np.sin(global_yaw)

    # 3. 速度转换（可选）：只对旋转部分进行转换（不考虑平移）
    local_vel = tensor[:, :, 4:6]  # (NA, T, 2)
    # 旋转矩阵由 center.heading 给出
    ch = center.heading
    rot = np.array([[np.cos(ch), -np.sin(ch)],
                    [np.sin(ch),  np.cos(ch)]])
    global_vel = np.einsum('ij,ntj->nti', rot, local_vel)

    # 构造转换后的 global_tensor
    global_tensor = tensor.copy()
    global_tensor[:, :, 0:2] = xy_global     # 更新中心位置
    global_tensor[:, :, 2] = global_cos      # 更新 cos(yaw)
    global_tensor[:, :, 3] = global_sin      # 更新 sin(yaw)
    global_tensor[:, :, 4:6] = global_vel     # 更新速度

    # --- 可视化 ---
    # 根据全局数据计算绘图范围
    all_x = global_tensor[:, :, 0]
    all_y = global_tensor[:, :, 1]
    margin = 10
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin

    frames = []

    for t in range(num_timesteps):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"Global Time step {t}")

        for i in range(num_vehicles):
            if validity[i, t] < 0.5:
                continue
            # 提取转换后全局下车辆状态
            x = global_tensor[i, t, 0]
            y = global_tensor[i, t, 1]
            cos_yaw = global_tensor[i, t, 2]
            sin_yaw = global_tensor[i, t, 3]
            length = global_tensor[i, t, 6]
            width = global_tensor[i, t, 7]
            yaw = np.arctan2(sin_yaw, cos_yaw)
            yaw_deg = np.degrees(yaw)

            half_length = length / 2
            half_width = width / 2
            # 计算左下角坐标（因为 Rectangle 以左下角为定位点）
            bottom_left_x = x - (half_length * np.cos(yaw) - half_width * np.sin(yaw))
            bottom_left_y = y - (half_length * np.sin(yaw) + half_width * np.cos(yaw))
            
            rect = patches.Rectangle(
                (bottom_left_x, bottom_left_y), 
                length, width, 
                angle=yaw_deg, 
                edgecolor='blue', 
                facecolor='none', 
                lw=2
            )
            ax.add_patch(rect)
            ax.plot(x, y, 'ro')  # 绘制车辆中心

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"Global GIF saved to {output_gif}")

# exp: visualize_scene_template(scene_template)
def visualize_scene_template(scene_template, output_gif="scene_template.gif", fps=2):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import imageio
    """
    可视化 scene_template 中的场景为 GIF 动画。
    
    输入的 scene_template 格式示例：
    
    scene_template = {
        "scene_metadata": {
            "log_name": "nuplan log name",
            "map_name": "nuplan map/city name, e.g. us-nv-las-vegas-strip",
            "initial_token": "The token I gave to you",
        },
        "frames": [
            {
                "token": <lidar_pc token>,
                "ego_status": {
                    "ego_pose": np.array([x, y, heading]),  # global 坐标系下，heading 单位为弧度
                    "ego_velocity": np.array([vx, vy]),       # global 坐标系下
                    "ego_acceleration": np.array([ax, ay]),   # global 坐标系下（此处可设为 0）
                },
                "annotations": {
                    "boxes": np.array([   # 每行表示一个目标车辆的 bbox，[x, y, z, l, w, h, heading]
                        [x, y, z, l, w, h, heading],
                        ...
                    ]),
                    "velocity_3d": np.array([   # 每行对应一个目标车辆的速度，[vx, vy, vz]（vz 设为 0）
                        [vx, vy, vz],
                        ...
                    ]),
                    "names": [ "vehicle", ... ],
                    "track_tokens": [ token, ... ]
                }
            },
            ...
        ]
    }
    
    所有的坐标信息均为 global 坐标系下的数值。
    """
    frames = scene_template["frames"]

    # 计算全局绘图范围（从 ego 和所有车辆的 bbox 中提取 x,y 范围）
    all_x = []
    all_y = []
    for frame in frames:
        # ego 的位置
        ego_pose = frame["ego_status"]["ego_pose"]
        all_x.append(ego_pose[0])
        all_y.append(ego_pose[1])
        # 每帧的目标车辆 bbox
        boxes = frame["annotations"]["boxes"]
        if boxes.size > 0:
            # boxes 为二维数组 (N,7)，取第 0 列和第 1 列
            all_x.extend(boxes[:, 0])
            all_y.extend(boxes[:, 1])
    margin = 10
    x_min = min(all_x) - margin
    x_max = max(all_x) + margin
    y_min = min(all_y) - margin
    y_max = max(all_y) + margin

    gif_frames = []
    for idx, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"Frame {idx}")

        # --- 绘制 ego 车辆 ---
        ego_status = frame["ego_status"]
        ego_pose = ego_status["ego_pose"]  # [x, y, heading]
        ego_x, ego_y, ego_heading = ego_pose
        ego_vel = ego_status["ego_velocity"]
        
        # 此处假设 ego 车辆尺寸未知，使用默认尺寸 (4,2) m 绘制 ego 车辆
        ego_length = 4.0
        ego_width = 2.0
        half_length = ego_length / 2.0
        half_width = ego_width / 2.0
        # 根据中心坐标及旋转角计算左下角坐标
        ego_bl_x = ego_x - (half_length * np.cos(ego_heading) - half_width * np.sin(ego_heading))
        ego_bl_y = ego_y - (half_length * np.sin(ego_heading) + half_width * np.cos(ego_heading))
        rect_ego = patches.Rectangle((ego_bl_x, ego_bl_y), ego_length, ego_width, 
                                     angle=np.degrees(ego_heading), 
                                     edgecolor="blue", facecolor="none", lw=2, label="ego")
        ax.add_patch(rect_ego)
        ax.plot(ego_x, ego_y, "bo")  # ego 中心点

        # 绘制 ego 速度向量（以箭头表示）
        ax.arrow(ego_x, ego_y, ego_vel[0], ego_vel[1],
                 head_width=0.5, head_length=0.5, fc="blue", ec="blue")

        # --- 绘制其他车辆 ---
        ann = frame["annotations"]
        boxes = ann["boxes"]
        velocities = ann["velocity_3d"]
        # 绘制每个目标车辆的 bbox
        if boxes.size > 0:
            for i in range(boxes.shape[0]):
                x, y, z, l, w, h, heading = boxes[i]
                half_l = l / 2.0
                half_w = w / 2.0
                # 计算左下角
                bl_x = x - (half_l * np.cos(heading) - half_w * np.sin(heading))
                bl_y = y - (half_l * np.sin(heading) + half_w * np.cos(heading))
                rect = patches.Rectangle((bl_x, bl_y), l, w, 
                                         angle=np.degrees(heading),
                                         edgecolor="red", facecolor="none", lw=2)
                ax.add_patch(rect)
                ax.plot(x, y, "ro")
                # 绘制速度向量（红色箭头）
                vx, vy, vz = velocities[i]
                ax.arrow(x, y, vx, vy, head_width=0.5, head_length=0.5, fc="red", ec="red")
        
        # 绘制当前帧的 lidar_pc token 作为标题的补充（可选）
        ax.text(x_min + 5, y_max - 5, f"Token: {frame['token']}", color="black", fontsize=12)
        
        # 将当前帧转换为图像
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(image)
        plt.close(fig)
    
    # 生成 GIF 动画
    imageio.mimsave(output_gif, gif_frames, fps=fps)
    print(f"GIF saved to {output_gif}")

def convert_scene_tensor_wrapper_to_scene_template(scene_tensor_wrapper):
    """
    将 scene_tensor_wrapper 中的 scene_tensor 从 local 坐标系转换为 global 坐标系，
    并构造出符合 scene_template 格式的字典，所有位置信息均保持在 global 坐标系下。

    注意：
      - ego车辆为 scene_tensor 中 idx==0 的 agent，直接使用转换后的 global 信息。
      - idx2token 记录了所有有效 agent（包括 ego）的 token，顺序与 scene_tensor 中有效 agent 顺序一致。
      - annotations 中仅包含非 ego agent（i>=1），并且仅加入存在 token 映射的 agent。

    输入的 scene_tensor_wrapper 数据结构:
        scene_tensor: SceneTensor 对象，包含：
            - tensor: np.ndarray, shape (NA, 21, 8)，每辆车在 21 帧下的状态，
                      各维度依次为 [x, y, cos(yaw), sin(yaw), vx, vy, length, width]
            - validity: np.ndarray, shape (NA, 21)，表示每辆车在各帧是否有效
        trans: 3x3 的 np.ndarray，将 local 转换到 global 的齐次变换矩阵
        center: 一个中心对象，需包含属性 heading（单位：弧度），用于角度转换
                （注意：ego 的 global 信息不通过 center 的 x,y,heading获取，而是使用 scene_tensor 中 idx==0 的数据）
        offset: np.ndarray, shape (2,)，表示在 global 坐标系下的 xy 偏移
        idx2token: List[str]，记录所有有效 agent 的 token（包括 ego）
        log_name: str
        scenario_token: str
        map_name: str
        lidar_pc: List[str]，长度为 21，对应 21 帧 lidar_pc token

    输出的 scene_template 格式:
        {
            scene_metadata: {
                log_name: ...,
                map_name: ...,
                initial_token: ...,
            },
            frames: [
                {
                    token: <lidar_pc token>,
                    ego_status: {
                        ego_pose: np.array([x, y, heading]),         # global 坐标系下（ego为 idx==0 的agent）
                        ego_velocity: np.array([vx, vy]),              # global 坐标系下
                        ego_acceleration: np.array([ax, ay]),          # global 坐标系下（暂设为 0）
                    },
                    annotations: {
                        boxes: np.array([
                            [x, y, z, l, w, h, heading],  # global 坐标系下，z, h 设为 0
                            ...
                        ]),
                        velocity_3d: np.array([
                            [vx, vy, vz],  # global 坐标系下，vz 设为 0
                            ...
                        ]),
                        names: [ "vehicle", ... ],
                        track_tokens: [ token, ... ]
                    }
                },
                ...
            ]
        }
    """
    # 解包 scene_tensor_wrapper 中的字段
    st = scene_tensor_wrapper.scene_tensor  # SceneTensor 对象
    trans = scene_tensor_wrapper.trans        # 3x3 local -> global
    # center 中仅使用 heading 进行角度转换
    center = scene_tensor_wrapper.center      # 假设具有 center.heading
    offset = scene_tensor_wrapper.offset      # global 坐标系下的 xy 偏移, shape (2,)
    idx2token = scene_tensor_wrapper.idx2token  # List[str]，记录所有有效 agent 的 token（包括 ego）
    log_name = scene_tensor_wrapper.log_name
    scenario_token = scene_tensor_wrapper.scenario_token
    map_name = scene_tensor_wrapper.map_name
    lidar_pc = scene_tensor_wrapper.lidar_pc    # List[str]，21 帧
    agent_type = scene_tensor_wrapper.agent_type 
    # 获取 scene_tensor 中的数据：local 坐标下的 tensor 和 validity
    tensor = decode_scene_tensor(st.tensor)  # shape: (NA, 21, 8)
    validity = st.validity   # shape: (NA, 21)
    num_vehicles, num_timesteps, _ = tensor.shape

    # --- 构造有效 agent 的 token 映射 ---
    # 记录所有在任意帧有效的 agent索引，顺序与 idx2token 一致
    valid_indices = [i for i in range(num_vehicles) if np.any(validity[i, :] >= 0.5)]
    # 假设 idx2token 的长度与 valid_indices 的数量一致
    agent_token_mapping = {i: idx2token[k] for k, i in enumerate(valid_indices)}

    # --- 1. local -> global 转换 ---
    # 位置转换：将 (x, y) 用齐次坐标变换并加上偏移
    xy_local = tensor[:, :, 0:2]      # (NA, 21, 2)
    ones = np.ones((num_vehicles, num_timesteps, 1))
    xy_local_h = np.concatenate([xy_local, ones], axis=-1)  # (NA, 21, 3)
    xy_global_h = np.einsum('ij,ntj->nti', trans, xy_local_h)
    xy_global = xy_global_h[:, :, :2] + offset
    # 更新 global_tensor 的 (x, y)
    global_tensor = tensor.copy()
    global_tensor[:, :, 0:2] = xy_global

    # 角度转换：local yaw = arctan2(sin, cos) + center.heading
    local_yaw = np.arctan2(tensor[:, :, 3], tensor[:, :, 2])
    global_yaw = local_yaw + center.heading
    global_tensor[:, :, 2] = np.cos(global_yaw)
    global_tensor[:, :, 3] = np.sin(global_yaw)

    # 速度转换：将 local 速度旋转到 global 坐标
    local_vel = tensor[:, :, 4:6]  # (NA, 21, 2)
    ch = center.heading
    rot = np.array([[np.cos(ch), -np.sin(ch)],
                    [np.sin(ch),  np.cos(ch)]])
    global_vel = np.einsum('ij,ntj->nti', rot, local_vel)
    global_tensor[:, :, 4:6] = global_vel

    # --- 2. 构造 scene_template ---
    frames_list = []
    for t in range(num_timesteps):
        # Ego 信息：直接使用 scene_tensor 中 idx==0 的数据（假设 ego 一定是有效的）
        ego_x = global_tensor[0, t, 0]
        ego_y = global_tensor[0, t, 1]
        ego_yaw = np.arctan2(global_tensor[0, t, 3], global_tensor[0, t, 2])
        ego_pose = np.array([ego_x, ego_y, ego_yaw])
        ego_velocity = global_tensor[0, t, 4:6]
        ego_acceleration = np.array([0.0, 0.0])  # 暂设为 0

        boxes = []
        velocities = []
        names = []
        track_tokens = []
        # annotations 中只包含非 ego agent（i>=1）且存在 token 映射的 agent
        for i in range(1, num_vehicles):
            if validity[i, t] < 0.5:
                continue
            if i not in agent_token_mapping:
                continue
            x = global_tensor[i, t, 0]
            y = global_tensor[i, t, 1]
            length = global_tensor[i, t, 6]
            width = global_tensor[i, t, 7]
            yaw = np.arctan2(global_tensor[i, t, 3], global_tensor[i, t, 2])
            # 构造 box: [x, y, z, l, w, h, heading]，z, h 暂设为 0
            box = [x, y, 0.0, length, width, 0.0, yaw]
            boxes.append(box)
            vx = global_tensor[i, t, 4]
            vy = global_tensor[i, t, 5]
            velocities.append([vx, vy, 0.0])
            names.append(agent_type[i])
            token = agent_token_mapping[i]
            track_tokens.append(token)
        
        frame_dict = dict(
            token=lidar_pc[t],
            ego_status=dict(
                ego_pose=ego_pose,         # 来自 idx==0 的全局信息
                ego_velocity=ego_velocity,
                ego_acceleration=ego_acceleration,
            ),
            annotations=dict(
                boxes = np.array(boxes) if boxes else np.empty((0, 7)),
                velocity_3d = np.array(velocities) if velocities else np.empty((0, 3)),
                names = names,
                track_tokens = track_tokens,
            )
        )
        frames_list.append(frame_dict)

    scene_template = dict(
        scene_metadata=dict(
            log_name=log_name,
            map_name=map_name,
            initial_token=scenario_token,
        ),
        frames=frames_list
    )

    return scene_template


def update_config_for_training(cfg: DictConfig) -> None:
    # Set the struct to False to allow changes to the config
    OmegaConf.set_struct(cfg, False)
    # You can change settings in cfg here

    # choose your scenario log name and lidar_pc token
    cfg.scenario_filter.log_names = ['log_1', 'log_2']
    cfg.scenario_filter.scenario_tokens = ['lidar_pc_1', 'lidar_pc_2']
    # specify the root directories for nuplan data and maps
    cfg.scenario_builder.scenario_mapping.data_root = 'NUPLAN_DATA_ROOT'
    cfg.scenario_builder.scenario_mapping.map_root = 'NUPLAN_MAP_ROOT'

    # Save all interpolations and remove keys that were only used for interpolation and have no further use.
    OmegaConf.resolve(cfg)
    # Freeze the config to avoid accidental changes
    OmegaConf.set_struct(cfg, True)
    return cfg

def build_scenarios_from_config(
    cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool
) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker) 

def get_features_from_scenario(builder: AbstractFeatureBuilder, scenario: AbstractScenario) -> SceneTensorWrapper:   
    vector_data=HorizonVectorFeatureBuilderV2.get_features_from_scenario(builder, scenario, iteration=0)
    return compute_features(builder, vector_data)

def compute_features(builder: AbstractFeatureBuilder, vector_data: HorizonVectorV2) -> SceneTensorWrapper:
    vehicles = vector_data.data["agents"]["VEHICLE"]  # type: ignore
    peds = vector_data.data["agents"]["PEDESTRIAN"]  # type: ignore
    bic = vector_data.data["agents"]["BICYCLE"]  # type: ignore
    ego = vector_data.data["ego"]  # t x 7 tensor
    track_token_id_mapping = vector_data.data['track_token_id_mapping']
    for key in track_token_id_mapping:
        track_token_id_mapping[key] = {int(v): k for k, v in track_token_id_mapping[key].items()}
    map_name = vector_data.data['map_name']
    log_name = vector_data.data['log_name']
    lidar_pc = vector_data.data['lidarpc']
    scenario_token = vector_data.data['scenario_token']
    offset = vector_data.data['offset']
    center = vector_data.data['center']
    trans = vector_data.data['trans']

    n_timesteps = len(vehicles)

    # get all unique track ids
    ids_veh = np.unique(vehicles[..., 0])
    ids_ped = np.unique(peds[..., 0])
    ids_bic = np.unique(bic[..., 0])
    idx2token = ['ego']
    agent_type = ['vehicle']
    # n_agents = len(ids_veh) + len(ids_ped) + len(ids_bic)
    # n_max_agents = sum(builder._num_max_agents)
    n_max_agents = 128
    # create nan tensor
    scene_tensor = np.zeros((n_max_agents, n_timesteps, N_SCENE_TENSOR_FEATURES))
    scene_tensor_validity = np.zeros((n_max_agents, n_timesteps))

    n_agents = 0
    for i, id_ in enumerate(ids_veh):
        if i >= builder._num_max_agents[0]:
            break
        mask = vehicles[..., 0] == id_
        time_validity_ = mask.sum(axis=1).astype(bool)
        if time_validity_.sum() < 5:
            continue
        scene_tensor[n_agents, time_validity_] = encode_agent(
            "VEHICLE",
            vehicles[mask],  # type: ignore
        )
        scene_tensor_validity[n_agents] = time_validity_
        idx2token.append(track_token_id_mapping['VEHICLE'][int(id_)])
        agent_type.append('vehicle')
        n_agents += 1

    for i, id_ in enumerate(ids_ped):
        if i >= builder._num_max_agents[1]:
            break
        mask = peds[..., 0] == id_
        time_validity_ = mask.sum(axis=1).astype(bool)
        if time_validity_.sum() < 5:
            continue

        scene_tensor[n_agents, time_validity_] = encode_agent(
            "PEDESTRIAN",
            peds[mask],  # type: ignore
        )
        scene_tensor_validity[n_agents] = time_validity_
        idx2token.append(track_token_id_mapping['PEDESTRIAN'][int(id_)])
        agent_type.append('pedestrian')
        n_agents += 1

    for i, id_ in enumerate(ids_bic):
        if i >= builder._num_max_agents[2]:
            break
        mask = bic[..., 0] == id_
        time_validity_ = mask.sum(axis=1).astype(bool)
        if time_validity_.sum() < 5:
            continue
        scene_tensor[n_agents, time_validity_] = encode_agent("BICYCLE", bic[mask])  # type: ignore
        scene_tensor_validity[n_agents] = time_validity_
        idx2token.append(track_token_id_mapping['BICYCLE'][int(id_)])
        agent_type.append('bicycle')
        n_agents += 1

    # append the ego state

    ego_state = encode_ego(ego)
    scene_tensor = np.concatenate(
        [np.expand_dims(ego_state, 0), scene_tensor], axis=0
    )
    # normalize the scene with x-mu/2sigma
    scene_tensor = (scene_tensor - np.array(FEATURE_MEANS)) / (
        2.0 * np.array(FEATURE_STD)
    )
    scene_tensor_validity = np.concatenate(
        [np.ones((1, n_timesteps)), scene_tensor_validity], axis=0
    )

    # encode road_graph
    road_graph = np.zeros(
        shape=(
            builder._map_num_max_polylines,
            builder._map_num_points_polylines,
            builder._map_dim_points_polyline,
        ),
        dtype=np.float32,
    )
    road_graph_validity = np.zeros_like(road_graph)
    map_data = vector_data.data["map"]

    for i in range(min(builder._map_num_max_polylines, len(map_data))):
        road_graph[i], road_graph_validity[i] = builder._encode_map_object(map_data[i])

    # use the same normalization as for the other coordinates
    road_graph[..., :2] = (road_graph[..., :2] - np.array(FEATURE_MEANS)[:2]) / (
        2 * np.array(FEATURE_STD)[:2]
    )

    scene_tensor = SceneTensor(
        tensor=scene_tensor.astype(np.float32),
        validity=scene_tensor_validity.astype(np.float32),
        road_graph=road_graph.astype(np.float32),
        road_graph_validity=road_graph_validity.astype(np.float32),
    )
    scene_tensor_wrapper = SceneTensorWrapper(
        scene_tensor=scene_tensor,
        trans=trans,
        center=center,
        offset=offset,
        idx2token=idx2token,
        log_name=log_name,
        scenario_token=scenario_token,
        map_name=map_name,
        lidar_pc=lidar_pc,
        agent_type=agent_type
    )
    return scene_tensor_wrapper
    # convert_scene_tensor_wrapper_to_scene_template(scene_tensor_wrapper)

def create_intent_attack_mask(scene_tensor_wrapper, n_chosen_prob=0.3):
    """
    针对意图攻击任务生成 task mask，同时对 scene_tensor_wrapper 的 tensor 进行修改：
      - 找到 ego (idx==0) 到其他 agent 第一个 timestep 距离最短的 agent，
        并交换 agent 1 与该 agent 的顺序。
      - 修改交换后的 agent 1 的最后一个 timestep 的前4个特征为 ego 的对应值，
        并确保其在最后 timestep 有效。
    """
    scene_tensor_features = scene_tensor_wrapper.scene_tensor
    task_mask = torch.zeros_like(scene_tensor_features.tensor)

    # 获取 ego 车在第一个 timestep 的位置
    ego_pos = scene_tensor_features.tensor[0, 0, :2]
    valid_mask = scene_tensor_features.validity[1:, 0]  # 取除 ego 之外的 agent

    # 计算其他 agent 在第五个 timestep 与 ego 的距离
    distances = torch.norm(scene_tensor_features.tensor[1:, 4, :2] - ego_pos, dim=1)
    distances = torch.where(valid_mask.bool(), distances, torch.tensor(1e6, dtype=torch.float32, device=scene_tensor_features.tensor.device))

    # 找到最近的 agent
    closest_index = torch.argmin(distances[:Counter(scene_tensor_wrapper.agent_type)['vehicle']-1]).item() + 1

    # 交换 agent 1 和 closest_index 对应的 agent 数据（包括 tensor 和 validity）
    temp_tensor = scene_tensor_features.tensor[1].clone()
    scene_tensor_features.tensor[1] = scene_tensor_features.tensor[closest_index]
    scene_tensor_features.tensor[closest_index] = temp_tensor

    temp_valid = scene_tensor_features.validity[1].clone()
    scene_tensor_features.validity[1] = scene_tensor_features.validity[closest_index]
    scene_tensor_features.validity[closest_index] = temp_valid

    # 更新 idx2token 和 agent_type：交换位置1与closest_index对应的条目
    if len(scene_tensor_wrapper.idx2token) > closest_index:
        scene_tensor_wrapper.idx2token[1], scene_tensor_wrapper.idx2token[closest_index] = (
            scene_tensor_wrapper.idx2token[closest_index],
            scene_tensor_wrapper.idx2token[1],
        )
    if len(scene_tensor_wrapper.agent_type) > closest_index:
        scene_tensor_wrapper.agent_type[1], scene_tensor_wrapper.agent_type[closest_index] = (
            scene_tensor_wrapper.agent_type[closest_index],
            scene_tensor_wrapper.agent_type[1],
        )

    # 应用 intent attack：将 agent 1 最后一个 timestep 的前4个特征置为 ego 对应值
    scene_tensor_features.tensor[1, -1, :4] = scene_tensor_features.tensor[0, -1, :4].clone()
    scene_tensor_features.validity[1, -1] = 1

    # 设置 task mask：前5个 timestep 全部置1，最后 timestep 的前2个 agent 置1
    task_mask[:, :5, :] = 1
    task_mask[:2, -1] = 1

    return task_mask

def create_task_mask(scene_tensor_wrapper, n_chosen_prob=0.3):
    """
    针对 scene_gen 任务生成 task mask：
      根据 agent 的有效性（任意 timestep 有效），按概率 mask 有效的 agent。
    """
    scene_tensor_features = scene_tensor_wrapper.scene_tensor
    n_valid_agents = scene_tensor_features.validity.any(dim=-1).sum().item()
    task_mask = torch.zeros_like(scene_tensor_features.tensor)

    mask = torch.rand(n_valid_agents, device=scene_tensor_features.tensor.device) < n_chosen_prob
    task_mask[:n_valid_agents] = mask.unsqueeze(-1).unsqueeze(-1)
    
    return task_mask

def unsqueeze(features: Dict):
    scene_tensor = features['scene_tensor']
    task_mask = features['task_mask']
    
    # Extract relevant tensors from scene_tensor
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    # Use unsqueeze to add an extra dimension at position 0
    tensor = tensor.unsqueeze(0)  # Correct method is unsqueeze
    validity = validity.unsqueeze(0)
    road_graph = road_graph.unsqueeze(0)
    road_graph_validity = road_graph_validity.unsqueeze(0)
    task_mask = task_mask.unsqueeze(0)

    return {
        'task_mask': task_mask,
        'scene_tensor': SceneTensor(tensor, validity, road_graph, road_graph_validity)
    }

def squeeze(features: Dict):
    scene_tensor = features['scene_tensor']
    task_mask = features['task_mask']
    
    # Extract relevant tensors from scene_tensor
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    # Use squeeze to remove dimensions of size 1
    tensor = tensor.squeeze(0)  # Removes the dimension at position 0 if it's size 1
    validity = validity.squeeze(0)
    road_graph = road_graph.squeeze(0)
    road_graph_validity = road_graph_validity.squeeze(0)
    task_mask = task_mask.squeeze(0)

    # Return the modified tensors in the same structure
    return {
        'task_mask': task_mask,
        'scene_tensor': SceneTensor(tensor, validity, road_graph, road_graph_validity)
    }

def generate_new_scene_tensor_wrapper_from_wrapper(scene_tensor_wrapper, model, task):
    """
    输入已生成的 scene_tensor_wrapper，依据 task（"bp", "scene_gen", "intent_attack"）
    构造对应的 task_mask，并生成新的 scene_tensor_wrapper。
    """
    features = {"scene_tensor": scene_tensor_wrapper.scene_tensor}
    
    if task == "bp":
        bp_mask = torch.zeros_like(scene_tensor_wrapper.scene_tensor.tensor)
        bp_mask[:, :5, :] = 1
        features["task_mask"] = bp_mask
    elif task == "scene_gen":
        features["task_mask"] = create_task_mask(scene_tensor_wrapper)
    elif task == "intent_attack":
        features["task_mask"] = create_intent_attack_mask(scene_tensor_wrapper)
    else:
        raise ValueError(f"Unknown task: {task}")

    with torch.no_grad(), autocast(enabled=False):
        model.eval()
        predictions = model.forward_inference(unsqueeze(features))
        model.train()
    predictions['scene_tensor']=features['scene_tensor']
    predictions['scene_tensor'].tensor = predictions['sampled_tensor']
    predictions = squeeze(predictions)
    # 更新 scene_tensor.tensor，同时保持其他字段不变
    new_scene_tensor_wrapper = scene_tensor_wrapper
    new_scene_tensor_wrapper.scene_tensor = predictions['scene_tensor']
    return new_scene_tensor_wrapper

def to_tensor(scene_tensor_wrapper):
    """
    将 SceneTensorWrapper 转换为 torch.Tensor 格式。
    """
    scene_tensor = scene_tensor_wrapper.scene_tensor
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    scene_tensor.tensor = torch.tensor(tensor)
    scene_tensor.validity = torch.tensor(validity)
    scene_tensor.road_graph = torch.tensor(road_graph)
    scene_tensor.road_graph_validity = torch.tensor(road_graph_validity)

    scene_tensor_wrapper.scene_tensor = scene_tensor
    return scene_tensor_wrapper

def to_numpy(scene_tensor_wrapper):
    """
    将 SceneTensorWrapper 转换为 numpy 格式。
    """
    scene_tensor = scene_tensor_wrapper.scene_tensor
    tensor = scene_tensor.tensor.numpy()
    validity = scene_tensor.validity.numpy()
    road_graph = scene_tensor.road_graph.numpy()
    road_graph_validity = scene_tensor.road_graph_validity.numpy()

    scene_tensor.tensor = tensor
    scene_tensor.validity = validity
    scene_tensor.road_graph = road_graph
    scene_tensor.road_graph_validity = road_graph_validity

    scene_tensor_wrapper.scene_tensor = scene_tensor
    return scene_tensor_wrapper
    
if __name__ == '__main__':
    
    with open('/cpfs01/user/yenaisheng/SceneGen/scripts/generation/config.yaml', 'r') as f:
        cfg = OmegaConf.load(f)
    # specify log_name, lidarpc_token here    
    # cfg = update_config_for_training(cfg)
    worker = build_worker(cfg)
    scenario_builder = build_scenario_builder(cfg)
    scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
    model = build_torch_module_wrapper(cfg.model)
    if cfg.checkpoint:
        my_ckpt = torch.load(cfg.checkpoint, map_location="cpu")
        my_ckpt["state_dict"] = {k[6:]: v for k, v in my_ckpt["state_dict"].items()}
        model.load_state_dict(my_ckpt["state_dict"], strict=True)
    
    feature_builder = model.get_list_of_required_feature()[0]

    for scenario in scenarios:
        scene_tensor_wrapper = get_features_from_scenario(feature_builder, scenario)
                
        # 可视化初始 scene_tensor
        visualize_scene_tensor(scene_tensor_wrapper.scene_tensor)

        # numpy 转换为 torch
        scene_tensor_wrapper = to_tensor(scene_tensor_wrapper)
        task = "intent_attack"  # 可选 "bp"、"scene_gen"、"intent_attack"
        new_scene_tensor_wrapper = generate_new_scene_tensor_wrapper_from_wrapper(scene_tensor_wrapper, model, task)
        
        # torch 转换为 numpy
        new_scene_tensor_wrapper = to_numpy(new_scene_tensor_wrapper)
        
        # 生成并可视化新的 scene_template
        new_scene_template = convert_scene_tensor_wrapper_to_scene_template(new_scene_tensor_wrapper)
        visualize_scene_template(new_scene_template)
