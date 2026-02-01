# 3D Object Detection With Sensor Fusion - Lidar + Camera Data

This project runs a LiDAR–camera fusion pipeline for 3D object detection: it uses YOLO on images, projects LiDAR points into the image, clusters them per object, fits 3D boxes with PCA, and evaluates against KITTI-style ground truth (precision, recall, F1, distance error). You get annotated images with predicted and ground-truth 3D boxes, plus text and plot outputs.

---

## Install

From the project folder, install dependencies:

```bash
pip install numpy opencv-python ultralytics open3d scikit-learn matplotlib tqdm
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

---

## Paths You Need to Set

Everything depends on four kinds of paths. Set them as below.

### 1. Dataset path (KITTI data)

**What it is:** The folder that contains your KITTI-format **training** data. Inside it you must have these four subfolders:

| Subfolder     | Contents                                  | Example files      |
| ------------- | ----------------------------------------- | ------------------ |
| `image_2/`  | Camera images                             | `000000.png`, … |
| `velodyne/` | LiDAR point clouds                        | `000000.bin`, … |
| `calib/`    | Calibration (P2, Tr_velo_to_cam, R0_rect) | `000000.txt`, … |
| `label_2/`  | Ground-truth labels                       | `000000.txt`, … |

Frame IDs must match across folders (e.g. `000000.png`, `000000.bin`, `000000.txt` in each).

**Where to set it:** You must give the path to your own KITTI data folder.

- **main.py:** Use the `--dataset` argument with your path, e.g.`--dataset /path/to/your/training`
- **PointscldEvaluation.py:** At the top of the file, set `data_root` to your path, e.g.
  `data_root = Path("/path/to/your/training")`

**Example paths (you can use your own):**

- **KITTI mini:**If you put KITTI mini inside this project folder, the path is:`kitti_mini/kitti_tiny_3D/training`So you run: `--dataset kitti_mini/kitti_tiny_3D/training`.If you put it somewhere else, use that folder path instead (e.g. `D:/datasets/kitti_mini/kitti_tiny_3D/training`).
- **Official KITTI (full):**Use the folder that contains `image_2/`, `velodyne/`, `calib/`, `label_2/`. Examples:

  - Windows: `G:/Data/KITTI/training` or `D:/kitti_object/training`
  - Mac/Linux: `/Users/me/data/kitti/training` or `/home/user/datasets/kitti/training`

  Run: `python main.py --dataset YOUR_PATH --results results/main_evaluation`
  (replace `YOUR_PATH` with your actual folder; in PointscldEvaluation.py set `data_root = Path("YOUR_PATH")`).

---

## Where to download the KITTI dataset

We use only two dataset options:

| Dataset                  | Size    | Where to download                                                            | Notes                                                                                                        |
| ------------------------ | ------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **KITTI mini**     | ~185 MB | [GitHub: kitti_mini](https://github.com/xuehaipiaoxiang/kitti_mini)             | Small subset. Clone or download ZIP; use folder `kitti_tiny_3D/training` or `kitti_mini/kitti/training`. |
| **Official KITTI** | ~41 GB  | [KITTI Object Detection](https://www.cvlibs.net/datasets/kitti/eval_object.php) | Full dataset. Create an account, then download left images, Velodyne, calib, and training labels.            |

After downloading, use a folder that contains `image_2/`, `velodyne/`, `calib/`, and `label_2/` as your **dataset path** (see above).

---

### 2. Results path (output folder)

**What it is:** The directory where the script writes all outputs (annotated images, evaluation text file, distance plot, and for PointscldEvaluation per-frame folders and CSVs).

**Where to set it:**

- **main.py:** Use the `--results` argument, e.g.`--results results/main_evaluation`
- **PointscldEvaluation.py:** At the top, set `out_root`, e.g.
  `out_root = _script_dir / "results" / "pointcloud_evaluation"`

If you don’t pass `--results` in main.py, it picks a default under `results/` depending on the dataset path.

---

### 3. Model path (YOLO weights)

**What it is:** Path to the YOLO segmentation model file (e.g. `yolo11s-seg.pt`). If the file is not found, Ultralytics will try to download it when you run the script.

**Where to set it:**

- **main.py:** Use the `--model` argument, e.g.`--model yolo11s-seg.pt`
- **PointscldEvaluation.py:** At the top, set `model_path`, e.g.
  `model_path = "yolo11s-seg.pt"`

Use a full path if the weights are in another folder (e.g. `C:/models/yolo11s-seg.pt`).

---

### 4. Project/script directory (where you run from)

**What it is:** The folder that contains `main.py` and `PointscldEvaluation.py`. Relative paths in the scripts (like `kitti_mini/kitti_tiny_3D/training`) are resolved from the **current working directory** when you run Python, so you should run the scripts from this project folder.

**What to do:** Open a terminal, go to the project folder, then run the script with **your dataset path**:

```bash
cd /path/to/3D-object-detection-sensor-fusion
python main.py --dataset YOUR_DATASET_PATH --results results/main_evaluation
```

Replace `YOUR_DATASET_PATH` with the folder that has your KITTI data (e.g. `kitti_mini/kitti_tiny_3D/training` if KITTI mini is inside the project, or `G:/Data/KITTI/training` for full KITTI). If you run from another directory, use absolute paths for `--dataset` and `--results`.

---

## Quick run (after install)

From the project folder, run with **your dataset path** (example below uses KITTI mini inside the project):

```bash
python main.py --dataset YOUR_DATASET_PATH --results results/main_evaluation --max_images 5
```

Example if KITTI mini is in the project folder: `--dataset kitti_mini/kitti_tiny_3D/training`. Otherwise use the path where your KITTI training data lives.

For point-cloud evaluation (PointscldEvaluation.py), set `data_root` and `out_root` at the top of the file as above, then:

```bash
python PointscldEvaluation.py
```

Outputs will appear under the **results path** you set (e.g. `results/main_evaluation/` or `results/pointcloud_evaluation/`).
