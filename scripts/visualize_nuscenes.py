#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--predictions", default="reports/bevformer_outputs.pt")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-predictions", type=int, default=80)
    return parser.parse_args()


def load_predictions(path):
    path = Path(path)
    if not path.exists():
        return None

    import torch

    outputs = torch.load(path, map_location="cpu")
    if not outputs or not outputs[0]:
        return None
    pts_bbox = outputs[0][0].get("pts_bbox")
    if not pts_bbox:
        return None
    boxes = pts_bbox["boxes_3d"].tensor.detach().cpu()
    scores = pts_bbox["scores_3d"].detach().cpu()
    labels = pts_bbox["labels_3d"].detach().cpu()
    return boxes, scores, labels


def rotated_box_xy(cx, cy, length, width, yaw):
    base = [
        (length / 2, width / 2),
        (length / 2, -width / 2),
        (-length / 2, -width / 2),
        (-length / 2, width / 2),
    ]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    points = []
    for x, y in base:
        points.append((cx + x * cos_yaw - y * sin_yaw, cy + x * sin_yaw + y * cos_yaw))
    points.append(points[0])
    return points


def main():
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=False)
    sample = nusc.sample[args.sample_index]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    first_cam_token = sample["data"]["CAM_FRONT"]
    first_cam_data = nusc.get("sample_data", first_cam_token)
    first_img = Image.open(data_root / first_cam_data["filename"]).convert("RGB")

    for ax, cam in zip(axes.flat, CAMERAS):
        sd = nusc.get("sample_data", sample["data"][cam])
        image = Image.open(data_root / sd["filename"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(cam)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "six_camera_view.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(first_img)
    _, boxes, camera_intrinsic = nusc.get_sample_data(first_cam_token)
    for box in boxes[:30]:
        corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
        ax.plot(corners[0, [0, 1, 2, 3, 0]], corners[1, [0, 1, 2, 3, 0]], color="lime", linewidth=1.5)
        ax.plot(corners[0, [4, 5, 6, 7, 4]], corners[1, [4, 5, 6, 7, 4]], color="lime", linewidth=1.5)
        for i, j in zip(range(4), range(4, 8)):
            ax.plot(corners[0, [i, j]], corners[1, [i, j]], color="lime", linewidth=1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "camera_boxes_overlay.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([0], [0], marker="^", s=120, color="tab:blue", label="ego")
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        x, y, _ = ann["translation"]
        w, l, _ = ann["size"]
        ax.add_patch(plt.Rectangle((x - l / 2, y - w / 2), l, w, fill=False, edgecolor="tab:green", linewidth=1.5))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "bev_gt_boxes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([0], [0], marker="^", s=120, color="tab:blue", label="ego")
    predictions = load_predictions(args.predictions)
    if predictions is None:
        raise RuntimeError(
            f"Real BEVFormer predictions are missing or invalid: {args.predictions}"
        )
    plotted = 0
    boxes, scores, labels = predictions
    order = scores.argsort(descending=True)
    colors = plt.cm.tab10
    for idx in order:
        score = float(scores[idx])
        if score < args.score_threshold or plotted >= args.max_predictions:
            continue
        box = boxes[idx]
        cx, cy = float(box[0]), float(box[1])
        dx, dy = float(box[3]), float(box[4])
        yaw = float(box[6])
        points = rotated_box_xy(cx, cy, dx, dy, yaw)
        xs, ys = zip(*points)
        label = int(labels[idx])
        ax.plot(xs, ys, color=colors(label % 10), linewidth=1.2, alpha=0.85)
        plotted += 1
    ax.set_title(f"BEVFormer predictions: {plotted} boxes")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "bev_pred_boxes.png", dpi=160)
    plt.close(fig)

    print(f"Wrote nuScenes figures to {out_dir}")


if __name__ == "__main__":
    main()
