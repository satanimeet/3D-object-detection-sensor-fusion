import time
import csv
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# CONFIG — set paths here or via script directory
_script_dir = Path(__file__).resolve().parent
data_root = _script_dir / "kitti_mini" / "kitti_tiny_3D" / "training"
model_path = "yolo11s-seg.pt"  # YOLO downloads if missing
out_root = _script_dir / "results" / "pointcloud_evaluation"
conf_thr = 0.35
mask_thr = 0.35


# class & label maps
coco_to_kitti = {
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "car": "Car",
    "motorcycle": "Cyclist",
    "bus": "Truck",
    "truck": "Truck",
    "van": "Truck",
    "train": "Tram",
    "airplane": "DontCare",
    "boat": "DontCare",
    "traffic light": "DontCare",
    "fire hydrant": "DontCare",
}
class_to_index_map = {"Car": 0, "Truck": 1, "Pedestrian": 2, "Cyclist": 3, "Tram": 4, "DontCare": 5}
gt_remap = {"van": "Truck", "bus": "Truck", "truck": "Truck", "person": "Pedestrian", "person_sitting": "Pedestrian", "misc": "DontCare"}
eval_classes = [n for n in class_to_index_map if n != "DontCare"]
class_names = sorted(eval_classes, key=lambda n: class_to_index_map[n])

# calibration and projection 
def load_calibration_file(path: Path):
    calibration_data = {}
    for line in path.read_text().splitlines():
        for key in ("P2", "Tr_velo_to_cam", "R0_rect"):
            if line.startswith(key):
                shape = (3, 4) if key != "R0_rect" else (3, 3)
                calibration_data[key] = np.array(line.split(maxsplit=1)[1].split(), dtype=np.float64).reshape(shape)
    return calibration_data

def project_lidar_to_image(points: np.ndarray, calibration_data: dict):
    lidar_xyz = points[:, :3]
    homogeneous_lidar = np.hstack((lidar_xyz, np.ones((lidar_xyz.shape[0], 1))))
    camera_points = homogeneous_lidar @ calibration_data["Tr_velo_to_cam"].T
    camera_points = camera_points[:, :3] @ calibration_data["R0_rect"].T
    forward_mask = camera_points[:, 2] > 0.1
    lidar_xyz = lidar_xyz[forward_mask]
    camera_points = camera_points[forward_mask]
    homogeneous_camera = np.hstack((camera_points, np.ones((camera_points.shape[0], 1))))
    projected = homogeneous_camera @ calibration_data["P2"].T
    pixel_uv = (projected[:, :2] / projected[:, 2:3]).astype(int)
    return lidar_xyz, pixel_uv

def compute_box_corners_camera(dimensions, location, rot_y):
    height, width, length = dimensions
    x_coords = [ length/2,  length/2, -length/2, -length/2,  length/2,  length/2, -length/2, -length/2 ]
    y_coords = [        0,         0,          0,          0,     -height,     -height,     -height,     -height ]
    z_coords = [  width/2, -width/2, -width/2,  width/2,    width/2,   -width/2,   -width/2,    width/2 ]
    rotation_y_matrix = np.array([
        [ np.cos(rot_y), 0, np.sin(rot_y)],
        [             0, 1,             0],
        [-np.sin(rot_y), 0, np.cos(rot_y)],
    ])
    corners_camera = rotation_y_matrix @ np.vstack([x_coords, y_coords, z_coords])
    corners_camera += np.array(location).reshape(3, 1)
    return corners_camera

def transform_camera_to_velodyne(corners_camera, Tr_velo_to_cam):
    homogeneous_cam = np.vstack((corners_camera, np.ones((1, 8))))
    T4 = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))
    return (np.linalg.inv(T4) @ homogeneous_cam)[:3]

def load_ground_truth_objects(label_path: Path):
    objects = []
    for line in label_path.read_text().splitlines():
        if not line:
            continue
        parts = line.split()
        raw = parts[0].lower()
        name = gt_remap.get(raw, raw)
        if name.lower() == "dontcare":
            continue
        dimensions = [float(parts[8]), float(parts[9]), float(parts[10])]
        location = [float(parts[11]), float(parts[12]), float(parts[13])]
        rotation_y = float(parts[14])
        objects.append({"type": name, "dimensions": dimensions, "location": location, "rotation_y": rotation_y})
    return objects

# Main evaluation loop

start_time = time.time()
out_root.mkdir(parents=True, exist_ok=True)
yolo_model = YOLO(model_path)

global_stats = defaultdict(lambda: dict(seg=0, in_gt=0, gt=0))
frame_count = 0

velodyne_dir = data_root / "velodyne"
image_dir = data_root / "image_2"
calib_dir = data_root / "calib"
label_dir = data_root / "label_2"

for vel_file in sorted(velodyne_dir.glob("*.bin")):
    frame_id = vel_file.stem
    frame_out = out_root / frame_id
    frame_out.mkdir(exist_ok=True)

    calibration_data = load_calibration_file(calib_dir / f"{frame_id}.txt")
    raw_lidar = np.fromfile(vel_file, np.float32).reshape(-1, 4)
    lidar_xyz, pixel_uv = project_lidar_to_image(raw_lidar, calibration_data)

    image_bgr = cv2.imread(str(image_dir / f"{frame_id}.png"))
    yolo_result = yolo_model.predict(image_bgr, conf=conf_thr, retina_masks=True)[0]
    overlay_image = yolo_result.plot()

    collected_points = []
    collected_class_ids = []
    if yolo_result.masks:
        for mask_tensor, det_box in zip(yolo_result.masks.data, yolo_result.boxes):
            if det_box.conf.item() <= conf_thr:
                continue
            coco_name = yolo_model.names[int(det_box.cls.item())].lower()
            kitti_name = coco_to_kitti.get(coco_name, "DontCare")
            if kitti_name == "DontCare":
                continue
            class_id = class_to_index_map[kitti_name]

            binary_mask = (mask_tensor.cpu().numpy() > mask_thr).astype(np.uint8)
            morph_kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, morph_kernel)
            binary_mask = cv2.erode(binary_mask, morph_kernel, iterations=2)

            mask_height, mask_width = binary_mask.shape
            uv_valid = (
                (pixel_uv[:, 0] >= 0) & (pixel_uv[:, 0] < mask_width) &
                (pixel_uv[:, 1] >= 0) & (pixel_uv[:, 1] < mask_height)
            )
            lidar_in_img = lidar_xyz[uv_valid]
            uv_sel = pixel_uv[uv_valid]
            hits = binary_mask[uv_sel[:, 1], uv_sel[:, 0]].astype(bool)
            pts_sel = lidar_in_img[hits]

            if pts_sel.size:
                collected_points.append(pts_sel)
                collected_class_ids += [class_id] * pts_sel.shape[0]

    if collected_points:
        candidate_points = np.vstack(collected_points)
        candidate_class_ids = np.array(collected_class_ids)
    else:
        candidate_points = np.empty((0, 3))
        candidate_class_ids = np.array([], dtype=int)

    if candidate_points.shape[0] > 0:
        dbscan_model = DBSCAN(eps=0.7, min_samples=15).fit(candidate_points[:, :3])
        labels, counts = np.unique(dbscan_model.labels_, return_counts=True)
        keep_labels = {lb for lb, c in zip(labels, counts) if lb != -1 and c >= 25}
        keep_mask = np.isin(dbscan_model.labels_, list(keep_labels))
        segmented_points = candidate_points[keep_mask]
        segmented_class_ids = candidate_class_ids[keep_mask]
    else:
        segmented_points = np.empty((0, 3))
        segmented_class_ids = np.array([], dtype=int)

    np.save(frame_out / f"{frame_id}_pts.npy", segmented_points)
    np.save(frame_out / f"{frame_id}_cls.npy", segmented_class_ids)
    cv2.imwrite(str(frame_out / f"{frame_id}_overlay.png"), overlay_image)

    gt_objects = load_ground_truth_objects(label_dir / f"{frame_id}.txt")
    transform_velo_to_cam = calibration_data["Tr_velo_to_cam"]
    raw_points_o3d = o3d.utility.Vector3dVector(raw_lidar[:, :3])

    frame_stats = defaultdict(lambda: dict(seg=0, in_gt=0, gt=0))

    for class_name, class_id in class_to_index_map.items():
        if class_name == "DontCare":
            continue
        n_seg = (segmented_class_ids == class_id).sum()
        frame_stats[class_name]["seg"] = n_seg
        global_stats[class_name]["seg"] += n_seg

    for obj in gt_objects:
        gt_name = obj["type"].capitalize()
        if gt_name not in class_to_index_map:
            continue

        corners_cam = compute_box_corners_camera(obj["dimensions"], obj["location"], obj["rotation_y"])
        corners_velo = transform_camera_to_velodyne(corners_cam, transform_velo_to_cam)
        oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners_velo.T))

        n_gt_raw = len(oriented_bbox.get_point_indices_within_bounding_box(raw_points_o3d))
        frame_stats[gt_name]["gt"] += n_gt_raw
        global_stats[gt_name]["gt"] += n_gt_raw

        seg_in_gt = oriented_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(segmented_points))
        n_in_gt = len(seg_in_gt)
        frame_stats[gt_name]["in_gt"] += n_in_gt
        global_stats[gt_name]["in_gt"] += n_in_gt

    with (frame_out / f"{frame_id}_metrics.csv").open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["Class", "SegPts", "InsideGT", "GTpts", "Prec%", "Rec%", "F1%"])
        for cname in eval_classes:
            s = frame_stats[cname]
            prec_pct = s["in_gt"] / s["seg"] * 100 if s["seg"] else 0
            rec_pct = s["in_gt"] / s["gt"] * 100 if s["gt"] else 0
            f1_pct = 2 * prec_pct * rec_pct / (prec_pct + rec_pct) if (prec_pct + rec_pct) > 0 else 0
            writer.writerow([cname, s["seg"], s["in_gt"], s["gt"], f"{prec_pct:.1f}", f"{rec_pct:.1f}", f"{f1_pct:.1f}"])

    frame_count += 1
    print(f"Processed {frame_id}")

print("\n=== Overall Segmentation Quality (Counts-Based) ===")
print("Class      SegPts  InsideGT  GTpts  Prec%  Rec%   F1%")
precisions, recalls, f1_scores = [], [], []
for cname in class_names:
    totals = global_stats[cname]
    prec_pct = totals["in_gt"] / totals["seg"] * 100 if totals["seg"] else 0
    rec_pct = totals["in_gt"] / totals["gt"] * 100 if totals["gt"] else 0
    f1_pct = 2 * prec_pct * rec_pct / (prec_pct + rec_pct) if (prec_pct + rec_pct) > 0 else 0
    print(f"{cname:<11}{totals['seg']:8d}{totals['in_gt']:9d}{totals['gt']:7d}{prec_pct:7.1f}{rec_pct:7.1f}{f1_pct:7.1f}")
    precisions.append(prec_pct / 100)
    recalls.append(rec_pct / 100)
    f1_scores.append(f1_pct / 100)
print()

with (out_root / "overall_metrics_counts.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "SegPts", "InsideGT", "GTpts", "Precision", "Recall", "F1"])
    for cname, p, r, f1 in zip(class_names, precisions, recalls, f1_scores):
        totals = global_stats[cname]
        writer.writerow([cname, totals["seg"], totals["in_gt"], totals["gt"], p, r, f1])

elapsed = time.time() - start_time
minutes, seconds = divmod(elapsed, 60)
hours, minutes = divmod(minutes, 60)
print(f"\nElapsed: {int(hours)}h {int(minutes)}m {seconds:04.1f}s")
print(f"Frames evaluated: {frame_count}")
