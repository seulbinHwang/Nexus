from enum import Enum
from typing import Optional, Tuple
from PIL import Image,ImageDraw
import io
import torch
import math
import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from numba import jit
from numba.core import types
from numba.typed import Dict
from math import sin, cos
from nuplan.planning.training.callbacks.utils.visualization_utils import _draw_trajectory
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

import torch
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor
import aggdraw

class Color(Enum):
    """
    Collection of colors for visualizing element in the occupancy map.
    """
    BACKGROUND: Tuple[float, float, float] = (0, 0, 0)
    VEHICLE: Tuple[float, float, float] = (255, 0, 0)
    PEDESTRIAN: Tuple[float, float, float] = (0, 0, 255)
    BICYCLE: Tuple[float, float, float] = (0, 255, 0)
    GENERIC_OBJECTS: Tuple[float, float, float] = (0, 255, 255)

    ROADMAP: Tuple[float, float, float] = (54, 67, 94)
    AGENTS: Tuple[float, float, float] = (113, 100, 222)
    EGO: Tuple[float, float, float] = (82, 86, 92)
    TARGET_TRAJECTORY: Tuple[float, float, float] = (61, 160, 179)
    PREDICTED_TRAJECTORY: Tuple[float, float, float] = (158, 63, 120)
    BASELINE_PATHS: Tuple[float, float, float] = (210, 220, 220)
    HEATMAP: Tuple[float, float, float] = (255, 255, 255)


LABEL_TO_COLOR = {
    0: Color.BACKGROUND,
    1: Color.VEHICLE,
    2: Color.PEDESTRIAN,
    3: Color.BICYCLE,
    4: Color.GENERIC_OBJECTS
}


def get_occupancy_as_rgb(occupancy: npt.NDArray[np.float32],
                         ) -> npt.NDArray[np.uint8]:
    """
    Convert occupancy map to RGB image according to the probability.
    """
    num_frames, height, width = occupancy.shape
    occupancy = (occupancy[..., None] * (255, 255, 255)).astype(
        np.uint8).transpose([0, 3, 1, 2])  # [num_frames, 3, height, width]
    return np.asarray(occupancy)


def get_raster_with_trajectories_as_rgb(
        raster: Raster,
        target_trajectory: Optional[Trajectory] = None,
        predicted_trajectory: Optional[Trajectory] = None,
        pixel_size: float = 0.5,
        ego_longitudinal_offset: Optional[float] = 0.0,
) -> npt.NDArray[np.uint8]:
    """
    Create an RGB images of the raster layers overlayed with predicted / ground truth trajectories

    :param raster: input raster to visualize
    :param target_trajectory: target (ground truth) trajectory to visualize
    :param predicted_trajectory: predicted trajectory to visualize
    :param background_color: desired color of the image's background
    :param roadmap_color: desired color of the map raster layer
    :param agents_color: desired color of the agents raster layer
    :param ego_color: desired color of the ego raster layer
    :param target_trajectory_color: desired color of the target trajectory
    :param predicted_trajectory_color: desired color of the predicted trajectory
    :param pixel_size: [m] size of pixel in meters
    :return: constructed RGB image
    """
    grid_shape = (raster.height, raster.width)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((*grid_shape, 3),
                                           Color.BACKGROUND.value,
                                           dtype=np.uint8)
    image[raster.data[..., 2, :, :][0] > 0] = Color.ROADMAP.value
    image[raster.data[..., 3, :, :][0] > 0] = Color.BASELINE_PATHS.value
    # squeeze to shape of W*H only
    image[raster.data[..., 1, :, :][0] > 0] = Color.AGENTS.value
    image[raster.data[..., 0, :, :][0] > 0] = Color.EGO.value

    # Draw predicted and target trajectories
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY,
                         pixel_size, 3, 2, ego_longitudinal_offset)
    if predicted_trajectory is not None:
        _draw_trajectory(image, predicted_trajectory,
                         Color.PREDICTED_TRAJECTORY, pixel_size, 3, 2,
                         ego_longitudinal_offset)

    return image


def get_heatmap_as_rgb(predicted_heatmap: Tensor,
                       threshold: float = 0.05) -> npt.NDArray[np.uint8]:
    """
    render heatmap object to bev heatmap.
    :param predicted_heatmap: the predicted heatmap tensor class
    :param threshold: score threshold for rendering
    :return image: rendered RGB image of shape (H, W, 3)
    """
    heatmap = torch.sigmoid(
        predicted_heatmap.data).cpu().detach().numpy().max(1)[0]
    grid_shape = heatmap.shape[-2:]
    image: npt.NDArray[np.uint8] = np.full((*grid_shape, 3),
                                           Color.BACKGROUND.value,
                                           dtype=np.uint8)
    if (heatmap > threshold).sum() > 0:
        image[heatmap > threshold, :] = (
            heatmap[heatmap > threshold][:, None] / 2 * np.array(
                Color.HEATMAP.value).astype(np.float32)).astype(np.uint8)
    return image


def _draw_trajectory(
        image: npt.NDArray[np.uint8],
        trajectory: Trajectory,
        color: Color,
        pixel_size: float,
        radius: int = 7,
        thickness: int = 3,
        ego_longitudinal_offset: float = 0.0,
) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2 + int(ego_longitudinal_offset * grid_height)
    center_y = grid_height // 2

    coords_x = (center_x - trajectory.numpy_position_x / pixel_size).astype(
        np.int32)
    coords_y = (center_y - trajectory.numpy_position_y / pixel_size).astype(
        np.int32)
    idxs = np.logical_and.reduce([
        0 <= coords_x, coords_x < grid_width, 0 <= coords_y,
        coords_y < grid_height
    ])
    coords_x = coords_x[idxs]
    coords_y = coords_y[idxs]

    for point in zip(coords_y, coords_x):
        cv2.circle(
            image, point, radius=radius, color=color.value, thickness=-1)

    for point_1, point_2 in zip(
            zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:],
                                                   coords_x[1:])):
        cv2.line(
            image, point_1, point_2, color=color.value, thickness=thickness)


@jit(nopython=True)
def convert_box(box, l_shift, w_dilate, l_dilate):
    x, y, w, l, heading_rad = box
    w += w_dilate
    l += l_dilate
    # Define the rectangle vertices with respect to the center of the bounding
    # box
    half_w, half_l = w / 2, l / 2
    rect_vertices = [
        (-half_w, -half_l + l_shift),
        (-half_w, half_l + l_shift),
        (half_w, half_l + l_shift),
        (half_w, -half_l + l_shift)
    ]

    # Rotate and translate vertices to the right location
    transformed_vertices = []
    for vertex in rect_vertices:
        vx, vy = vertex
        rot_x = vx * cos(heading_rad) - vy * sin(heading_rad)
        rot_y = vx * sin(heading_rad) + vy * cos(heading_rad)
        transformed_vertices.append((rot_x + x, rot_y + y))
    return transformed_vertices


def fuse_images(background: np.array, foreground: np.array) -> np.array:
    '''
    Fuse two images together.
    background: [H, W, 4], RGBA image, [0-255]
    foreground: [H, W, 4], RGBA image, [0-255]
    return:
    fused_image: [H, W, 4], RGBA image, [0-255]
    '''
    # Normalize alpha channels to [0, 1]
    back_alpha = background[..., -1] / 255.0
    fore_alpha = foreground[..., -1] / 255.0

    # Fuse the RGB channels

    background[..., :-1] = (1 - fore_alpha[..., None]) * background[..., :-1] + fore_alpha[..., None] * foreground[..., :-1]

    # Fuse the alpha channel
    background[..., -1] = (1 - fore_alpha) * back_alpha + fore_alpha

    # Rescale the final alpha channel back to [0, 255]
    background[..., -1] = np.clip(background[..., -1] * 255, 0, 255)

    return background.astype(np.uint8)

def draw_bev_bboxes(box, canvas, color, fill=False,
                    l_shift=0.0, w_dilate=0.0, l_dilate=0.0):
    '''
    Draw a bounding box (BEV) on the canvas with optional transparency.
    
    Parameters:
    - box: List containing the [x, y, w, l, heading] values for the bounding box.
    - canvas: The image (canvas) on which to draw the bounding box.
    - color: The RGBA color used for the bounding box.
    - fill: Whether to fill the box (True) or just draw the outline (False).
    - l_shift: Length shift for the bounding box.
    - w_dilate: Width dilation for the bounding box.
    - l_dilate: Length dilation for the bounding box.
    
    Returns:
    - canvas: The updated canvas with the bounding box drawn.
    '''

    # Convert the box elements to float values
    box = [float(b) for b in box]
    
    # Transform the box vertices (using a helper function, `convert_box`)
    transformed_vertices = convert_box(box, l_shift, w_dilate, l_dilate)

    overlay = np.zeros((canvas.shape[0], canvas.shape[1], 3), dtype=np.uint8)

    # If filling the box, use the transparent fill method
    if False:
        # Fill the polygon (rectangle) with the RGBA color
        cv2.fillPoly(overlay, [np.array(transformed_vertices, dtype=np.int32)], color[:3])
    else:
        # Draw the polygon (rectangle) outline with the RGBA color
        cv2.polylines(overlay,
                      [np.array(transformed_vertices, dtype=np.int32)],
                      isClosed=True,
                      color=color[:3],
                      thickness=2)

    # Draw the vehicle's front (a triangle using the front and side centers)
    front_center_x = (transformed_vertices[0][0] + transformed_vertices[3][0]) / 2
    front_center_y = (transformed_vertices[0][1] + transformed_vertices[3][1]) / 2

    left_center_x = (transformed_vertices[0][0] + transformed_vertices[1][0]) / 2
    left_center_y = (transformed_vertices[0][1] + transformed_vertices[1][1]) / 2
    right_center_x = (transformed_vertices[2][0] + transformed_vertices[3][0]) / 2
    right_center_y = (transformed_vertices[2][1] + transformed_vertices[3][1]) / 2

    pts=[(int(front_center_x), int(front_center_y)),
         (int(left_center_x), int(left_center_y)),
         (int(right_center_x), int(right_center_y))]

    cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], isClosed=True, color=color[:3], thickness=2)
    
    opacity = np.any(overlay != 0, axis=-1) * color[3]
    opacity = np.expand_dims(opacity, axis=-1)
    # Set the alpha channel of the overlay to the specified transparency
    overlay = np.concatenate([overlay,opacity], axis=-1)

    # Fuse the overlay with the original canvas using the alpha transparency
    canvas = fuse_images(canvas, overlay)

    return canvas.astype(np.uint8)

class Draw:
    def __init__(self):
        pass
    @staticmethod
    def draw_bev_dot(box,canvas,color,fill=False,l_shift=0.0,w_dilate=0.0,l_dilate=0.0):
        # Convert the box elements to float values
        box = [float(b) for b in box]
        for b in box:
            if math.isnan(b):
                return canvas
        if box[2] < 0.1 or box[3] < 0.1:
            return canvas
        # Transform the box vertices (using a helper function, `convert_box`)
        transformed_vertices = convert_box(box, l_shift, w_dilate, l_dilate)

        # Calculate the center of the bounding box
        center_x = (transformed_vertices[0][0] + transformed_vertices[2][0]) / 2
        center_y = (transformed_vertices[0][1] + transformed_vertices[2][1]) / 2
        
        # The radius will be based on the minimum of width (w) or length (l), as a dot size
        radius = abs(min(box[2], box[3]))  # The radius should be based on the bounding box's width or length
        radius = max(int(radius/2), 1)  # Ensure a minimum radius of 1

        # the 4 point to locate circle
        pts=[(int(center_x-radius), int(center_y-radius)),(int(center_x+radius), int(center_y+radius))]
        overlay = Image.new('RGBA',canvas.size,(0,0,0,0))
        draw=ImageDraw.Draw(overlay)
        draw.ellipse(pts,fill=color)
        canvas = Image.alpha_composite(canvas,overlay)
        return canvas
    @staticmethod
    def draw_bev_bboxes(box, canvas, color, fill=False,
                        l_shift=0.0, w_dilate=0.0, l_dilate=0.0):
        '''
        Draw a bounding box (BEV) on the canvas with optional transparency.
        
        Parameters:
        - box: List containing the [x, y, w, l, heading] values for the bounding box.
        - canvas: The image (canvas) on which to draw the bounding box.
        - color: The RGBA color used for the bounding box.
        - fill: Whether to fill the box (True) or just draw the outline (False).
        - l_shift: Length shift for the bounding box.
        - w_dilate: Width dilation for the bounding box.
        - l_dilate: Length dilation for the bounding box.
        
        Returns:
        - canvas: The updated canvas with the bounding box drawn.
        '''
        # Convert the box elements to float values
        box = [float(b) for b in box]
        for b in box:
            if math.isnan(b):
                return canvas
        if box[2] < 0.1 or box[3] < 0.1:
            return canvas
        # Transform the box vertices (using a helper function, `convert_box`)
        transformed_vertices = convert_box(box, l_shift, w_dilate, l_dilate)

        # draw=ImageDraw.Draw(canvas)
        draw=aggdraw.Draw(canvas)
        pen = aggdraw.Pen(color, width=2)
        # draw.polygon(transformed_vertices,outline=color,width=2)
        flat_pts = [coord for point in transformed_vertices for coord in point]
        draw.polygon(flat_pts, pen)
        draw.flush()

        # Draw the vehicle's front (a triangle using the front and side centers)
        front_center_x = (transformed_vertices[0][0] + transformed_vertices[3][0]) / 2
        front_center_y = (transformed_vertices[0][1] + transformed_vertices[3][1]) / 2

        left_center_x = (transformed_vertices[0][0] + transformed_vertices[1][0]) / 2
        left_center_y = (transformed_vertices[0][1] + transformed_vertices[1][1]) / 2
        right_center_x = (transformed_vertices[2][0] + transformed_vertices[3][0]) / 2
        right_center_y = (transformed_vertices[2][1] + transformed_vertices[3][1]) / 2

        # pts=[(int(front_center_x), int(front_center_y)),
        #     (int(left_center_x), int(left_center_y)),
        #     (int(right_center_x), int(right_center_y))]
        # pts=[(front_center_x, front_center_y),
        #     (left_center_x, left_center_y),
        #     (right_center_x, right_center_y)]
        
        center_x = (left_center_x + right_center_x) / 2
        center_y = (left_center_y + right_center_y) / 2
        pts = [
            (front_center_x, front_center_y),
            (center_x, center_y),
        ]

        pts = [coord for point in pts for coord in point]

        # draw.polygon(pts,outline=color,width=2)
        pen = aggdraw.Pen(color, width=2)
        # draw.polygon(pts, pen)
        draw.line(pts, pen)
        draw.flush()
        return canvas
    @staticmethod
    def fuse_images(background,foreground):
        return Image.alpha_composite(background,foreground)
    @staticmethod
    def generate_gif(images,fps):
        gifs = []
        for gif,f in zip(images,fps):
            gif_io = io.BytesIO()
            gif[0].save(gif_io, format='GIF', append_images=gif[1:], save_all=True, duration=1000/f, loop=0)
            gif_io.seek(0)
            gifs.append(gif_io)
        return gifs

class Draw_bev_dot_interplate:
    def __init__(self,N=2):
        self.N = N # interpolation multiple
        self.env_agents_store = {} # env_name: {agent_idx: [box,color]}
    def draw_interplate(self,box,canvas,color,env_name,agent_idx,fill=False,l_shift=0.0,w_dilate=0.0,l_dilate=0.0):
        if env_name not in self.env_agents_store:
            self.env_agents_store[env_name] = {agent_idx:[box,color]}
        else:
            if agent_idx not in self.env_agents_store[env_name]:
                self.env_agents_store[env_name][agent_idx] = [box,color]
            else:
                origin_box= self.env_agents_store[env_name][agent_idx][0]
                origin_color = self.env_agents_store[env_name][agent_idx][1]
                for i in range(1,self.N):
                    inter_box = [box[j] * i / self.N + origin_box[j] * (self.N - i) / self.N for j in range(len(box))]
                    inter_color = [color[j] * i / self.N + origin_color[j] * (self.N - i) / self.N for j in range(len(color))]
                    canvas = draw_bev_dot(inter_box, canvas, inter_color,fill,l_shift,w_dilate,l_dilate)
                self.env_agents_store[env_name][agent_idx] = [box,color]

        return draw_bev_dot(box, canvas, color,fill,l_shift,w_dilate,l_dilate)

def draw_bev_dot(box, canvas, color, fill=False, 
                 l_shift=0.0, w_dilate=0.0, l_dilate=0.0):
    '''
    Draw a dot at the center of a bounding box (BEV) on the canvas with optional transparency.
    
    Parameters:
    - box: List containing the [x, y, w, l, heading] values for the bounding box.
    - canvas: The image (canvas) on which to draw the dot.
    - color: The RGBA color used for the dot.
    - fill: Whether to fill the box (True) or just draw the outline (False).
    - l_shift: Length shift for the bounding box.
    - w_dilate: Width dilation for the bounding box.
    - l_dilate: Length dilation for the bounding box.
    
    Returns:
    - canvas: The updated canvas with the dot drawn.
    '''

    # Convert the box elements to float values
    box = [float(b) for b in box]
    
    # Transform the box vertices (using a helper function, `convert_box`)
    transformed_vertices = convert_box(box, l_shift, w_dilate, l_dilate)

    # Calculate the center of the bounding box
    center_x = (transformed_vertices[0][0] + transformed_vertices[2][0]) / 2
    center_y = (transformed_vertices[0][1] + transformed_vertices[2][1]) / 2
    
    # The radius will be based on the minimum of width (w) or length (l), as a dot size
    radius = abs(min(box[2], box[3]))  # The radius should be based on the bounding box's width or length
    radius = max(int(radius/2), 1)  # Ensure a minimum radius of 1
    
    # Create a transparent overlay for drawing the dot
    overlay = np.zeros((canvas.shape[0], canvas.shape[1], 3), dtype=np.uint8)

    # Draw a filled circle (dot) at the center of the bounding box
    cv2.circle(overlay, (int(center_x), int(center_y)), int(radius), color[:-1], -1)
    opacity = np.any(overlay != 0, axis=-1) * color[3]
    opacity = np.expand_dims(opacity, axis=-1)
    # Set the alpha channel of the overlay to the specified transparency
    overlay = np.concatenate([overlay,opacity], axis=-1)

    # Combine the overlay with the original canvas using transparency
    canvas = fuse_images(canvas, overlay)

    return canvas.astype(np.uint8)

def generate_gif(images,fps):
    '''
    Generate a gif from a list of images.
    
    Parameters:
    - images: List of images to include in the gif.
    - fps: Frames per second for the gif.
    
    Returns:
    - gif: The gif object.
    '''
    def single_gif(image, f):
        '''
        - image: [N, H, W, C], C is 4 for RGBA, has been normalized to [0,1], numpy.ndarray
        - f: fps (frames per second)
        
        return: gif object in memory
        '''
        # Create a list to hold the frames
        frames = []
        
        # Convert each frame into an image object and append it to the frames list
        for frame in image:
            # Convert the normalized RGBA frame to [0, 255] for Pillow
            # frame = (frame * 255).astype(np.uint8)  # Convert to 8-bit (0-255 range)
            
            # Create a Pillow image from the numpy array
            pil_image = Image.fromarray(frame, 'RGBA')
            
            # Append the Pillow image to the frames list
            frames.append(pil_image)

        # Create a BytesIO buffer to hold the GIF in memory
        gif_io = io.BytesIO()
        
        # Save the frames as a GIF to the BytesIO buffer
        frames[0].save(gif_io, format='GIF', append_images=frames[1:], save_all=True, duration=1000/f, loop=0)
        
        # Seek back to the beginning of the GIF in the buffer
        gif_io.seek(0)
        
        # Return the GIF object (in-memory)
        return gif_io
    
    gif = []
    for image,f in zip(images,fps):
        gif.append(single_gif(image,f))
    return gif

def draw_bev_trajectory(trajectory, canvas, color, l_shift=0.0):
    if len(trajectory.shape) == 2:
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            cv2.line(canvas, (int(p1[0]), int(p1[1] + l_shift)), (int(p2[0]), int(p2[1] + l_shift)), color, thickness=1)
    else:
        for j in range(trajectory.shape[0]):
            for i in range(len(trajectory[j]) - 1):
                p1 = trajectory[j, i]
                p2 = trajectory[j, i + 1]
                cv2.line(canvas, (int(p1[0]), int(p1[1] + l_shift)), (int(p2[0]), int(p2[1] + l_shift)), color, thickness=1)
    return canvas


def map_to_rgb(tensor, colormap):
    """
    Convert a tensor using the provided colormap.
    :param tensor: tensor of shape (B, C, H, W).
    :param colormap: tensor of shape (C, 3).
    :return: tensor of shape (B, 3, H, W).
    """
    # B x C x H x W -> B x H x W x C
    tensor = tensor.permute(0, 2, 3, 1)

    # B x H x W x 3
    rgb_tensor = torch.matmul(tensor, colormap)

    # Clip values to [0, 1]
    rgb_tensor = torch.clamp(rgb_tensor, 0, 1)

    # B x 3 x H x W
    return rgb_tensor.permute(0, 3, 1, 2)

def draw_velocity(vx, vy, target_canvas, color):
    """
    Draw velocity vector on canvas.
    :param vx: x component of velocity.
    :param vy: y component of velocity.
    :param target_canvas: canvas to draw on.
    :param color: color of the velocity vector.
    :return: canvas with velocity vector drawn on it.
    """
    target_canvas = cv2.arrowedLine(target_canvas, (int(vx[0]), int(vy[0])), (int(vx[1]), int(vy[1])), color, 1)
    return target_canvas

def project_box_on_image(image, boxes, projection_mat, color, thickness):
    """
    Projects 3D bounding boxes onto a 2D image plane.

    Args:
        image (np.array): Image array of shape (H, W, 3).
        boxes (np.array): Array of shape (N, 7) representing 3D boxes [x, y, z, w, l, h, heading].
        projection_mat (np.array): Projection matrix of shape (4, 4).
        color (tuple): Color in (R, G, B) format for drawing boxes.
    
    Returns:
        np.array: Image with projected 3D bounding boxes.
    """
    image = np.ascontiguousarray(image, dtype=np.uint8)
    H, W, _ = image.shape

    for box in boxes:
        x, y, z, l, w, h, heading = box

        # Define the eight corners of the 3D bounding box
        corners = np.array([
            [-l / 2, -w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, w / 2, -h / 2],
            [-l / 2, w / 2, h / 2],
            [l / 2, -w / 2, -h / 2],
            [l / 2, -w / 2, h / 2],
            [l / 2, w / 2, -h / 2],
            [l / 2, w / 2, h / 2]
        ])
        corners[:, :3] += np.array([x, y, z])
        corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
        # Apply the rotation matrix
        rotation_mat = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])
        corners[:, :3] = corners[:, :3].dot(rotation_mat.T)

        # Project the 3D corners onto the 2D image plane
        projected_corners = corners.dot(projection_mat.T)

        projected_corners[:, 2] = np.clip(projected_corners[:, 2], a_min=1e-5, a_max=1e5)
        # Check if any corners are behind the camera
        if np.any(projected_corners[:, 2] <= 0):
            continue

        projected_corners = projected_corners[:, :2] / projected_corners[:, 2:3]

        # Convert to pixel coordinates
        projected_corners = projected_corners[:, :2]

        # Check if all corners are within image bounds
        if np.any(projected_corners[:, 0] < 0) or np.any(projected_corners[:, 0] >= W) or \
           np.any(projected_corners[:, 1] < 0) or np.any(projected_corners[:, 1] >= H):
            continue

        # Draw the bounding box edges on the image
        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        for edge in edges:
            pt1 = tuple(projected_corners[edge[0]].astype(int))
            pt2 = tuple(projected_corners[edge[1]].astype(int))
            cv2.line(image, pt1, pt2, color, thickness=thickness)

    return image

