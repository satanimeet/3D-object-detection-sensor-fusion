import numpy as np
import cv2
from ultralytics import YOLO
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import os
import time
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt

# -------------------------------
# Label and Color Mappings
# -------------------------------
coco_to_kitti = {
    'person':       'Pedestrian',
    'bicycle':      'Cyclist',
    'car':          'Car',
    'motorcycle':   'Cyclist',
    'bus':          'Truck',
    'truck':        'Truck',
    'van':          'Truck',
    'train':        'Tram',
    'airplane':     'DontCare',
    'boat':         'DontCare',
    'traffic light':'DontCare',
    'fire hydrant':'DontCare',
}

class_to_index_map = {
    'Car':        0,
    'Truck':      1,
    'Pedestrian': 2,
    'Cyclist':    3,
    'Tram':       4,
    'DontCare':   5
}

class_colors_rgb = {
    0: (255,   0,   0),   # Red
    1: (255,   0, 255),   # Magenta
    2: (  0,   0, 255),   # Blue
    3: (  0, 255,   0),   # Green
    4: (  0, 255, 255),   # Cyan
    5: (127, 127, 127)    # Gray
}

standard_dimensions = {
    0: [1.64, 1.52, 3.86],  # Car
    1: [2.90, 2.50, 8.50],   # Truck
    2: [1.70, 0.60, 0.80],   # Pedestrian
    3: [1.70, 0.75, 1.85],   # Cyclist
    4: [3.40, 2.50, 12.00]   # Tram
}

ground_truth_remap = {
    'van':            'Truck',
    'bus':            'Truck',
    'misc':           'DontCare',
    'person_sitting': 'Pedestrian'
}

# -------------------------------
# Ground Truth Utilities
# -------------------------------
def load_ground_truth_objects(label_file_path):
    objects = []
    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            raw_label = parts[0].lower()
            object_class = ground_truth_remap.get(raw_label, parts[0])
            if object_class == 'DontCare':
                continue
            dimensions = [float(parts[8]), float(parts[9]), float(parts[10])]
            location   = [float(parts[11]), float(parts[12]), float(parts[13])]
            rotation_y = float(parts[14])
            objects.append({
                'type':       object_class,
                'dimensions': dimensions,
                'location':   location,
                'rotation_y': rotation_y
            })
    return objects

def compute_3d_corners_for_object(dimensions, location, rotation_y):
    height, width, length = dimensions
    x_corners = [ length/2,  length/2, -length/2, -length/2,
                  length/2,  length/2, -length/2, -length/2 ]
    y_corners = [        0,         0,          0,          0,
                  -height,  -height,   -height,   -height   ]
    z_corners = [ width/2, -width/2, -width/2,  width/2,
                  width/2, -width/2, -width/2,  width/2  ]
    rotation_matrix = np.array([
        [ np.cos(rotation_y), 0, np.sin(rotation_y)],
        [                  0, 1,                 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    corners_3d = rotation_matrix @ np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0] += location[0]
    corners_3d[1] += location[1]
    corners_3d[2] += location[2]
    return corners_3d

def transform_camera_corners_to_lidar(corners_camera_3d, transformation_matrix):
    transformation_4x4 = np.vstack((transformation_matrix, [0, 0, 0, 1]))
    homogeneous_camera_corners = np.vstack((corners_camera_3d, np.ones((1, corners_camera_3d.shape[1]))))
    corners_lidar = np.linalg.inv(transformation_4x4) @ homogeneous_camera_corners
    return corners_lidar[:3]

def load_ground_truth_corner_boxes(calibration_file_path, label_file_path):
    calibration_data = {}
    with open(calibration_file_path, 'r') as file:
        for line in file:
            for key in ('P2', 'Tr_velo_to_cam', 'R0_rect'):
                if line.startswith(key):
                    values = np.array(line.split()[1:], dtype=np.float32)
                    shape  = (3,4) if key != 'R0_rect' else (3,3)
                    calibration_data[key] = values.reshape(shape)
    ground_truth_objects = load_ground_truth_objects(label_file_path)
    ground_truth_corner_boxes = []
    ground_truth_classes = []
    for obj in ground_truth_objects:
        corners_camera = compute_3d_corners_for_object(
            obj['dimensions'], obj['location'], obj['rotation_y']
        )
        corners_lidar = transform_camera_corners_to_lidar(
            corners_camera, calibration_data['Tr_velo_to_cam']
        )
        ground_truth_corner_boxes.append(corners_lidar)
        ground_truth_classes.append(obj['type'])
    return ground_truth_corner_boxes, ground_truth_classes

# -------------------------------
# 3D IoU Utilities
# -------------------------------
def convert_corners_to_axis_aligned_box(corners_3d):
    x_min = np.min(corners_3d[0]); x_max = np.max(corners_3d[0])
    y_min = np.min(corners_3d[1]); y_max = np.max(corners_3d[1])
    z_min = np.min(corners_3d[2]); z_max = np.max(corners_3d[2])
    return {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'z_min': z_min, 'z_max': z_max
    }

def compute_3d_iou(box_a, box_b):
    x_overlap = max(0, min(box_a['x_max'], box_b['x_max']) - max(box_a['x_min'], box_b['x_min']))
    y_overlap = max(0, min(box_a['y_max'], box_b['y_max']) - max(box_a['y_min'], box_b['y_min']))
    z_overlap = max(0, min(box_a['z_max'], box_b['z_max']) - max(box_a['z_min'], box_b['z_min']))
    intersection_volume = x_overlap * y_overlap * z_overlap

    volume_a = ((box_a['x_max'] - box_a['x_min']) *
                (box_a['y_max'] - box_a['y_min']) *
                (box_a['z_max'] - box_a['z_min']))
    volume_b = ((box_b['x_max'] - box_b['x_min']) *
                (box_b['y_max'] - box_b['y_min']) *
                (box_b['z_max'] - box_b['z_min']))
    union_volume = volume_a + volume_b - intersection_volume
    if union_volume == 0:
        return 0.0
    return intersection_volume / union_volume

# -------------------------------
# Projection Utilities
# -------------------------------
def project_3dbox_to_2d(corners_3d, projection_matrix, rectification_matrix, velo_to_cam_matrix):
    corners_3d_hom = np.vstack((corners_3d, np.ones((1, corners_3d.shape[1]))))
    corners_cam = velo_to_cam_matrix @ corners_3d_hom
    corners_rect = rectification_matrix @ corners_cam[:3, :]
    corners_rect_hom = np.vstack((corners_rect, np.ones((1, corners_rect.shape[1]))))
    image_points = projection_matrix @ corners_rect_hom
    image_points = image_points[:2, :] / image_points[2, :]
    return image_points.T

def reorder_box_corners(corners_3d_array: np.ndarray) -> np.ndarray:
    indices_sorted_by_z = np.argsort(corners_3d_array[:, 2])
    bottom_face_points  = corners_3d_array[indices_sorted_by_z[:4]]
    top_face_points     = corners_3d_array[indices_sorted_by_z[4:]]

    def sort_clockwise(points: np.ndarray) -> np.ndarray:
        centroid = points.mean(axis=0)
        angles   = np.arctan2(points[:, 1] - centroid[1],
                              points[:, 0] - centroid[0])
        return points[np.argsort(angles)]

    bottom_sorted_clockwise = sort_clockwise(bottom_face_points)
    top_sorted_clockwise    = sort_clockwise(top_face_points)
    return np.vstack((bottom_sorted_clockwise, top_sorted_clockwise))

def draw_3dbox_on_image(
    frame: np.ndarray,
    box_corners_2d: np.ndarray,
    line_color: Tuple[int, int, int] = (0, 0, 255),
    line_thickness: int = 2,
    iou_value: float | None = None,
    class_name: str | None = None,
) -> np.ndarray:
    box_corners_2d = box_corners_2d.astype(int)

    edge_index_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for start_idx, end_idx in edge_index_pairs:
        cv2.line(
            frame,
            tuple(box_corners_2d[start_idx]),
            tuple(box_corners_2d[end_idx]),
            line_color,
            line_thickness,
            lineType=cv2.LINE_AA,
        )

    x_min, y_min = box_corners_2d.min(0)
    if class_name is not None:
        cv2.putText(
            frame,
            class_name,
            (x_min, max(20, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    if iou_value is not None:
        cv2.putText(
            frame,
            f"IoU: {iou_value:.2f}",
            (x_min, max(40, y_min + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return frame

# -------------------------------
# Lidar-Camera Fusion & Segmentation
# -------------------------------
class LidarCameraFusion:
    def __init__(self, model_file_path,
                 confidence_threshold=0.35, mask_threshold=0.35):
        self.segmentation_model   = YOLO(model_file_path)
        self.confidence_threshold = confidence_threshold
        self.mask_threshold       = mask_threshold

    def _load_calibration(self, calibration_file_path):
        calibration = {}
        with open(calibration_file_path, 'r') as file:
            for line in file:
                for key in ('P2', 'Tr_velo_to_cam', 'R0_rect'):
                    if line.startswith(key):
                        values = np.array(line.split()[1:], dtype=np.float32)
                        shape  = (3,4) if key != 'R0_rect' else (3,3)
                        calibration[key] = values.reshape(shape)
        return calibration

    def project_lidar_to_image(self, raw_lidar_points, calibration_data):
        homogeneous_lidar = np.hstack((raw_lidar_points[:, :3],
                                       np.ones((raw_lidar_points.shape[0], 1))))
        camera_points = (homogeneous_lidar @ calibration_data['Tr_velo_to_cam'].T)[:, :3]
        camera_points = camera_points @ calibration_data['R0_rect'].T
        forward_mask  = camera_points[:, 2] > 0.1
        camera_points = camera_points[forward_mask]
        filtered_lidar = raw_lidar_points[forward_mask]
        homogeneous_camera = np.hstack((camera_points,
                                        np.ones((camera_points.shape[0], 1))))
        pixel_coordinates = (homogeneous_camera @ calibration_data['P2'].T)
        pixel_coordinates = (pixel_coordinates[:, :2] / pixel_coordinates[:, 2:3]).astype(int)
        return filtered_lidar, pixel_coordinates

    def process_frame(self, lidar_file_path, image_file_path, calibration_file_path):
        calibration_data = self._load_calibration(calibration_file_path)
        raw_lidar_points = np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4)
        image             = cv2.imread(image_file_path)
        lidar_points, pixel_coordinates = self.project_lidar_to_image(raw_lidar_points, calibration_data)

        segmentation_result = self.segmentation_model.predict(
            image, conf=self.confidence_threshold, retina_masks=True
        )[0]
        segmentation_overlay = segmentation_result.plot()

        point_groups             = []
        pixel_groups             = []
        group_class_identifiers  = []

        if segmentation_result.masks:
            for mask_tensor, detection_box in zip(
                    segmentation_result.masks.data,
                    segmentation_result.boxes
            ):
                if detection_box.conf.item() < self.confidence_threshold:
                    continue
                coco_label = self.segmentation_model.names[
                    int(detection_box.cls.item())
                ].lower()
                kitti_label = coco_to_kitti.get(coco_label, 'DontCare')
                if kitti_label == 'DontCare':
                    continue
                class_identifier = class_to_index_map[kitti_label]

                binary_mask = (mask_tensor.cpu().numpy() > self.mask_threshold).astype(np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                binary_mask = cv2.erode(binary_mask, kernel, iterations=2)

                mask_height, mask_width = binary_mask.shape
                valid_projection_mask = (
                    (pixel_coordinates[:,0] >= 0) &
                    (pixel_coordinates[:,0] < mask_width) &
                    (pixel_coordinates[:,1] >= 0) &
                    (pixel_coordinates[:,1] < mask_height)
                )
                valid_pixels = pixel_coordinates[valid_projection_mask]
                mask_hits    = binary_mask[
                    valid_pixels[:,1],
                    valid_pixels[:,0]
                ].astype(bool)
                selected_lidar_points = lidar_points[valid_projection_mask][mask_hits]
                selected_pixel_coords = valid_pixels[mask_hits]
                
                if selected_lidar_points.size:
                    point_groups.append(selected_lidar_points)
                    pixel_groups.append(selected_pixel_coords)
                    group_class_identifiers += [class_identifier] * selected_lidar_points.shape[0]

        if point_groups:
            aggregated_points = np.vstack(point_groups)
            aggregated_pixels = np.vstack(pixel_groups)
            aggregated_class_identifiers = np.array(group_class_identifiers)

            if aggregated_points.shape[0] > 0:
                clustering_result = DBSCAN(eps=0.7, min_samples=15).fit(
                    aggregated_points[:, :3]
                )
                cluster_labels, cluster_counts = np.unique(
                    clustering_result.labels_, return_counts=True
                )
                valid_clusters = [ 
                    label for label, count in zip(cluster_labels, cluster_counts)
                    if label != -1 and count >=25
                ]
                valid_mask = np.isin(clustering_result.labels_, valid_clusters)
                final_lidar_points       = aggregated_points[valid_mask]
                final_pixel_coords       = aggregated_pixels[valid_mask]
                final_class_identifiers  = aggregated_class_identifiers[valid_mask]
                final_cluster_labels     = clustering_result.labels_[valid_mask]
            else:
                final_lidar_points      = np.empty((0,4))
                final_pixel_coords      = np.empty((0,2))
                final_class_identifiers = np.array([], int)
                final_cluster_labels    = np.array([], int)
        else:
            final_lidar_points      = np.empty((0,4))
            final_pixel_coords      = np.empty((0,2))
            final_class_identifiers = np.array([], int)
            final_cluster_labels    = np.array([], int)

        return (final_lidar_points, final_class_identifiers, 
                final_cluster_labels, segmentation_overlay, 
                final_pixel_coords, image, calibration_data)

# -------------------------------
# Centroid-Based Cluster Merging
# -------------------------------
def merge_close_clusters(cluster_points, cluster_classes, distance_threshold=1.0):
    centroids = []
    for points in cluster_points:
        centroids.append(np.mean(points[:, :3], axis=0))
    centroids = np.array(centroids)
    
    n_clusters = len(cluster_points)
    graph = {i: [] for i in range(n_clusters)}
    
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if cluster_classes[i] != cluster_classes[j]:
                continue
            distance = np.linalg.norm(centroids[i] - centroids[j])
            if distance < distance_threshold:
                graph[i].append(j)
                graph[j].append(i)
    
    visited = [False] * n_clusters
    merged_clusters = []
    merged_classes = []
    
    for i in range(n_clusters):
        if visited[i]:
            continue
        current_indices = [i]
        stack = [i]
        visited[i] = True
        
        while stack:
            node = stack.pop()
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
                    current_indices.append(neighbor)
        
        merged_points = np.vstack([cluster_points[idx] for idx in current_indices])
        merged_class = cluster_classes[i]
        merged_clusters.append(merged_points)
        merged_classes.append(merged_class)
    
    return merged_clusters, merged_classes

# -------------------------------
# Dimension Adjustment Functions (Car only)
# -------------------------------
def adjust_box_dimensions(box, cluster_points, class_id):
    if class_id != 0:
        return box
    
    std_height, std_width, std_length = standard_dimensions[0]
    current_extent = box.extent
    current_length, current_width, current_height = current_extent
    
    length_ratio = current_length / std_length
    width_ratio = current_width / std_width
    height_ratio = current_height / std_height
    
    if length_ratio >= 0.8 and width_ratio >= 0.8 and height_ratio >= 0.8:
        return box
    
    local_points = (box.R.T @ (cluster_points[:, :3] - box.center).T).T
    
    min_vals = np.min(local_points, axis=0)
    max_vals = np.max(local_points, axis=0)
    
    height_extension = (std_height - (max_vals[2] - min_vals[2])) / 2
    min_vals[2] -= height_extension
    max_vals[2] += height_extension
    
    width_extension = (std_width - (max_vals[1] - min_vals[1])) / 2
    min_vals[1] -= width_extension
    max_vals[1] += width_extension
    
    sensor_to_center = box.center
    box_to_sensor = -sensor_to_center
    dot_product = box_to_sensor @ box.R[:,0]
    
    if dot_product > 0:
        min_vals[0] -= std_length - (max_vals[0] - min_vals[0])
    else:
        max_vals[0] += std_length - (max_vals[0] - min_vals[0])
    
    new_extent = max_vals - min_vals
    center_offset = (min_vals + max_vals) / 2
    
    new_box = o3d.geometry.OrientedBoundingBox()
    new_box.center = box.center + box.R @ center_offset
    new_box.extent = new_extent
    new_box.R = box.R
    return new_box

# -------------------------------
# PCA-Based Oriented Bounding Boxes
# -------------------------------
def create_pca_oriented_boxes(
    lidar_points, class_identifiers, cluster_labels
):
    cluster_points = []
    cluster_classes = []
    
    for cluster_label in np.unique(cluster_labels):
        if cluster_label == -1:
            continue
        cluster_mask = (cluster_labels == cluster_label)
        cluster_pts = lidar_points[cluster_mask]
        cluster_cls = class_identifiers[cluster_mask][0]
        if cluster_pts.shape[0] >= 25:
            cluster_points.append(cluster_pts)
            cluster_classes.append(cluster_cls)
    
    if cluster_points:
        merged_clusters, merged_classes = merge_close_clusters(
            cluster_points, cluster_classes
        )
    else:
        merged_clusters, merged_classes = [], []
    
    predicted_oriented_boxes = []
    predicted_point_clouds   = []
    predicted_class_names    = []
    
    for cluster_points, class_identifier in zip(merged_clusters, merged_classes):
        class_name = [
            name for name, idx in class_to_index_map.items()
            if idx == class_identifier
        ][0]

        pca_model = PCA(n_components=2)
        pca_model.fit(cluster_points[:, :2])
        principal_direction = pca_model.components_[0]
        yaw_angle = np.arctan2(principal_direction[1], principal_direction[0])

        rotation_to_local = np.array([
            [ np.cos(-yaw_angle), -np.sin(-yaw_angle)],
            [ np.sin(-yaw_angle),  np.cos(-yaw_angle)]
        ])
        rotated_xy = (rotation_to_local @ cluster_points[:, :2].T).T

        min_xy_local = rotated_xy.min(axis=0)
        max_xy_local = rotated_xy.max(axis=0)
        center_xy_local = (min_xy_local + max_xy_local) / 2

        min_z = cluster_points[:,2].min()
        max_z = cluster_points[:,2].max()
        center_z = (min_z + max_z) / 2

        center_xy_world = (rotation_to_local.T @ center_xy_local).T
        box_center = [center_xy_world[0], center_xy_world[1], center_z]
        box_size = [max_xy_local[0] - min_xy_local[0],
                    max_xy_local[1] - min_xy_local[1],
                    max_z - min_z]

        oriented_box = o3d.geometry.OrientedBoundingBox()
        oriented_box.center = box_center
        oriented_box.extent = box_size
        rotation_matrix_3d = np.eye(3)
        rotation_matrix_3d[:2, :2] = rotation_to_local.T
        oriented_box.R = rotation_matrix_3d
        oriented_box.color = (1, 0, 0)

        if class_identifier == 0:
            oriented_box = adjust_box_dimensions(oriented_box, cluster_points, class_identifier)

        point_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cluster_points[:, :3])
        )
        class_color = np.array(class_colors_rgb[class_identifier]) / 255.0
        point_cloud.paint_uniform_color(class_color)

        predicted_oriented_boxes.append(oriented_box)
        predicted_point_clouds.append(point_cloud)
        predicted_class_names.append(class_name)

    return predicted_oriented_boxes, predicted_point_clouds, predicted_class_names

# -------------------------------
# Evaluation Metrics with PR Curves
# -------------------------------
class EvaluationMetrics:
    def __init__(self, classes, iou_threshold=0.01):
        self.classes = classes
        self.iou_threshold = iou_threshold
        self.reset()
        
    def reset(self):
        self.tp = {cls: 0 for cls in self.classes}
        self.fp = {cls: 0 for cls in self.classes}
        self.fn = {cls: 0 for cls in self.classes}
        self.distance_error_sum = {cls: 0.0 for cls in self.classes}
        self.distance_error_count = {cls: 0 for cls in self.classes}
        self.detected_distances = {cls: [] for cls in self.classes}
        self.gt_distances = {cls: [] for cls in self.classes}
        
        # For PR curves - store predictions with their IoU scores and binary labels
        self.prediction_scores = {cls: [] for cls in self.classes}
        self.prediction_labels = {cls: [] for cls in self.classes}
        
    def update(self, pred_boxes, pred_classes, gt_boxes, gt_classes, pred_centers, gt_centers):
        matched_gt = [False] * len(gt_boxes)
        matched_pred = [False] * len(pred_boxes)
        
        # For each prediction, find best matching ground truth and compute IoU
        for j, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
            if pred_cls not in self.classes:
                continue
                
            best_iou = 0.0
            best_gt_idx = -1
            
            for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if matched_gt[i] or gt_cls != pred_cls:
                    continue
                iou = compute_3d_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Store prediction with IoU score and whether it's a true positive
            self.prediction_scores[pred_cls].append(best_iou)
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                self.prediction_labels[pred_cls].append(1)  # True positive
                matched_gt[best_gt_idx] = True
                matched_pred[j] = True
                
                # Update metrics
                self.tp[pred_cls] += 1
                gt_center = gt_centers[best_gt_idx]
                pred_center = pred_centers[j]
                error = np.linalg.norm(gt_center - pred_center)
                self.distance_error_sum[pred_cls] += float(error)
                self.distance_error_count[pred_cls] += 1
                self.detected_distances[pred_cls].append(float(np.linalg.norm(pred_center)))
                self.gt_distances[pred_cls].append(float(np.linalg.norm(gt_center)))
            else:
                self.prediction_labels[pred_cls].append(0)  # False positive
        
        # Count false positives and false negatives
        for j, matched in enumerate(matched_pred):
            if not matched and pred_classes[j] in self.classes:
                self.fp[pred_classes[j]] += 1
                
        for i, matched in enumerate(matched_gt):
            if not matched and gt_classes[i] in self.classes:
                self.fn[gt_classes[i]] += 1
                
    def calculate_metrics(self):
        metrics = {}
        for cls in self.classes:
            tp = self.tp[cls]
            fp = self.fp[cls]
            fn = self.fn[cls]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            mean_error = (self.distance_error_sum[cls] / self.distance_error_count[cls]
                          if self.distance_error_count[cls] > 0 else 0)
            metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'mean_distance_error': mean_error
            }
        return metrics

    def plot_distance_errors(self, save_path=None):
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        axes = axes.flatten()
        for idx, cls in enumerate(self.classes):
            ax = axes[idx]
            det = self.detected_distances[cls]
            gt = self.gt_distances[cls]
            if len(det) > 0 and len(gt) > 0:
                det = np.array(det)
                gt = np.array(gt)
                ax.scatter(det, gt, s=30, alpha=0.7, label='Detected Points')
                m = max(det.max(), gt.max())
                ax.plot([0, m], [0, m], color='red', linewidth=2, label='Ideal Line')
                mean_err = (self.distance_error_sum[cls] / self.distance_error_count[cls]
                            if self.distance_error_count[cls] > 0 else 0.0)
                ax.set_title(f"{cls} — Detected vs Ground Truth (Mean Err: {mean_err:.2f} m)")
                ax.set_xlabel("Detected Distance")
                ax.set_ylabel("Ground Truth Distance")
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_aspect('equal', adjustable='box')
            else:
                ax.set_title(f"{cls} — No Matches")
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        axes[-1].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


# -------------------------------
# Create 2D Image with 3D Boxes and IoU
# -------------------------------
def create_image_with_3d_boxes_and_iou(
    original_image,
    calibration_data,
    predicted_oriented_boxes,
    predicted_class_names,
    ground_truth_corner_boxes,
    ground_truth_classes
):
    image_with_boxes = original_image.copy()
    P2 = calibration_data['P2']
    R0_rect = calibration_data['R0_rect']
    Tr_velo_to_cam = calibration_data['Tr_velo_to_cam']
    velo_to_cam_4x4 = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    
    for corners_3d_gt, gt_class in zip(ground_truth_corner_boxes, ground_truth_classes):
        gt_projected_2d = project_3dbox_to_2d(corners_3d_gt, P2, R0_rect, velo_to_cam_4x4)
        image_with_boxes = draw_3dbox_on_image(image_with_boxes, gt_projected_2d, (0, 255, 0))
    
    for box, class_name in zip(predicted_oriented_boxes, predicted_class_names):
        corners_3d_pred = np.asarray(box.get_box_points()).T
        corners_3d_pred = reorder_box_corners(corners_3d_pred.T).T
        projected_corners_2d = project_3dbox_to_2d(corners_3d_pred, P2, R0_rect, velo_to_cam_4x4)
        aabb_pred = convert_corners_to_axis_aligned_box(corners_3d_pred)
        best_iou = 0.0
        for gt_corners in ground_truth_corner_boxes:
            aabb_gt = convert_corners_to_axis_aligned_box(gt_corners)
            iou_value = compute_3d_iou(aabb_pred, aabb_gt) 
            best_iou = max(best_iou, iou_value)
        image_with_boxes = draw_3dbox_on_image(
            image_with_boxes, projected_corners_2d, (0, 0, 255), iou_value=best_iou, class_name=class_name
        )
    return image_with_boxes

# -------------------------------
# Main Evaluation Function
# -------------------------------
def evaluate_pipeline(dataset_path, results_path, model_path, classes_to_evaluate, iou_threshold=0.01, max_images=None):
    os.makedirs(results_path, exist_ok=True)
    metrics = EvaluationMetrics(classes_to_evaluate, iou_threshold)
    fusion_model = LidarCameraFusion(model_path)
    image_dir = os.path.join(dataset_path, "image_2")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    if max_images is not None:
        image_files = image_files[:max_images]
    pbar = tqdm(image_files, desc="Evaluating images")
    
    for img_file in pbar:
        identifier = os.path.splitext(img_file)[0]
        pbar.set_postfix({"Image": identifier})
        lidar_path = os.path.join(dataset_path, "velodyne", identifier + ".bin")
        image_path = os.path.join(dataset_path, "image_2", img_file)
        calib_path = os.path.join(dataset_path, "calib", identifier + ".txt")
        label_path = os.path.join(dataset_path, "label_2", identifier + ".txt")
        try:
            (lidar_points, class_identifiers, cluster_labels, 
             segmentation_overlay, pixel_coords, original_image, calibration_data) = fusion_model.process_frame(
                lidar_path, image_path, calib_path
            )
            predicted_oriented_boxes, _, predicted_class_names = create_pca_oriented_boxes(
                lidar_points, class_identifiers, cluster_labels
            )
            ground_truth_corner_boxes, ground_truth_classes = load_ground_truth_corner_boxes(
                calib_path, label_path
            )
            pred_boxes_aabb = []
            pred_classes = []
            pred_centers = []
            for box, cls_name in zip(predicted_oriented_boxes, predicted_class_names):
                if cls_name in classes_to_evaluate:
                    corners = np.asarray(box.get_box_points()).T
                    pred_boxes_aabb.append(convert_corners_to_axis_aligned_box(corners))
                    pred_classes.append(cls_name)
                    pred_centers.append(np.array(box.center, dtype=float))
            pred_centers = np.array(pred_centers) if len(pred_centers) > 0 else np.empty((0,3))
            
            gt_boxes_aabb = []
            gt_classes = []
            gt_centers = []
            for corners, cls_name in zip(ground_truth_corner_boxes, ground_truth_classes):
                if cls_name in classes_to_evaluate:
                    gt_boxes_aabb.append(convert_corners_to_axis_aligned_box(corners))
                    gt_classes.append(cls_name)
                    gt_centers.append(np.mean(corners.T, axis=0))
            gt_centers = np.array(gt_centers) if len(gt_centers) > 0 else np.empty((0,3))
            
            metrics.update(pred_boxes_aabb, pred_classes, gt_boxes_aabb, gt_classes, pred_centers, gt_centers)
            
            result_image = create_image_with_3d_boxes_and_iou(
                original_image, calibration_data,
                predicted_oriented_boxes, predicted_class_names,
                ground_truth_corner_boxes, ground_truth_classes
            )
            output_path = os.path.join(results_path, f"{identifier}_iou.jpg")
            cv2.imwrite(output_path, result_image)
        except Exception as e:
            print(f"Error processing {identifier}: {str(e)}")
            continue
    return metrics.calculate_metrics(), metrics

# -------------------------------
# Run Evaluation
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LiDAR-Camera fusion 3D object detection evaluation")
    parser.add_argument("--dataset", type=str, default="G:/EM/Sem-7 Thesis/code/Data/training",
                        help="Path to KITTI-format training folder (image_2, velodyne, calib, label_2)")
    parser.add_argument("--results", type=str, default=None,
                        help="Output folder for results (default: results/kitti_mini_test when using kitti_mini, else results/iou_dbscan_evaluation)")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt", help="Path to YOLO segmentation model")
    parser.add_argument("--max_images", type=int, default=None, help="Max number of images to process (default: all)")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    RESULTS_PATH = args.results
    if RESULTS_PATH is None:
        RESULTS_PATH = "results/iou_dbscan_evaluation" if "kitti_mini" not in DATASET_PATH else "results/kitti_mini_test"
    MODEL_PATH = args.model
    CLASSES_TO_EVALUATE = ['Car', 'Truck', 'Pedestrian', 'Cyclist', 'Tram']
    IOU_THRESHOLD = 0.01
    MAX_IMAGES = args.max_images

    print("Starting evaluation...")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Results: {RESULTS_PATH}")
    print(f"  Max images: {MAX_IMAGES or 'all'}")
    start_time = time.time()
    
    metrics, metrics_obj = evaluate_pipeline(
        dataset_path=DATASET_PATH,
        results_path=RESULTS_PATH,
        model_path=MODEL_PATH,
        classes_to_evaluate=CLASSES_TO_EVALUATE,
        iou_threshold=IOU_THRESHOLD,
        max_images=MAX_IMAGES
    )
    
    print("\nEvaluation Results:")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'MeanDistErr':<13} {'TP':<6} {'FP':<6} {'FN':<6}")
    for cls, data in metrics.items():
        print(f"{cls:<12} {data['precision']:.4f}     {data['recall']:.4f}     {data['f1']:.4f}      "
              f"{data['mean_distance_error']:.4f}      {data['tp']:<6} {data['fp']:<6} {data['fn']:<6}")
    
    total_tp = sum(data['tp'] for data in metrics.values())
    total_fp = sum(data['fp'] for data in metrics.values())
    total_fn = sum(data['fn'] for data in metrics.values())
    total_dist_sum = sum(metrics_obj.distance_error_sum.get(cls, 0.0) for cls in CLASSES_TO_EVALUATE)
    total_dist_cnt = sum(metrics_obj.distance_error_count.get(cls, 0) for cls in CLASSES_TO_EVALUATE)
    overall_mean_dist_err = (total_dist_sum / total_dist_cnt) if total_dist_cnt > 0 else 0.0
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("\nOverall Metrics:")
    print(f"Precision:       {overall_precision:.4f}")
    print(f"Recall:          {overall_recall:.4f}")
    print(f"F1 Score:        {overall_f1:.4f}")
    print(f"MeanDistErr:     {overall_mean_dist_err:.4f}")
    print(f"Total TP:        {total_tp}")
    print(f"Total FP:        {total_fp}")
    print(f"Total FN:        {total_fn}")
    
    # Save results to file
    results_file = os.path.join(RESULTS_PATH, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'MeanDistErr':<13} {'TP':<6} {'FP':<6} {'FN':<6}\n")
        for cls, data in metrics.items():
            f.write(f"{cls:<12} {data['precision']:.4f}     {data['recall']:.4f}     {data['f1']:.4f}      "
                    f"{data['mean_distance_error']:.4f}      {data['tp']:<6} {data['fp']:<6} {data['fn']:<6}\n")
        
        f.write("\nOverall Metrics:\n")
        f.write(f"Precision:       {overall_precision:.4f}\n")
        f.write(f"Recall:          {overall_recall:.4f}\n")
        f.write(f"F1 Score:        {overall_f1:.4f}\n")
        f.write(f"MeanDistErr:     {overall_mean_dist_err:.4f}\n")
        f.write(f"Total TP:        {total_tp}\n")
        f.write(f"Total FP:        {total_fp}\n")
        f.write(f"Total FN:        {total_fn}\n")
    
    # Save distance plots
    plot_path = os.path.join(RESULTS_PATH, "detected_vs_gt_distances_per_class.png")
    metrics_obj.plot_distance_errors(save_path=plot_path)
    
    print(f"\nEvaluation completed in {time.time()-start_time:.2f} seconds")
    print(f"Results saved to: {results_file}")
    print(f"Distance plot saved to: {plot_path}")
    print(f"IoU images saved to: {RESULTS_PATH}")
