#!/usr/bin/env python3
import argparse
from pathlib import Path


REQUIRED_JSON = [
    "sample.json",
    "sample_data.json",
    "calibrated_sensor.json",
    "ego_pose.json",
    "sample_annotation.json",
    "scene.json",
    "category.json",
    "attribute.json",
]

REQUIRED_METADATA = [
    "nuscenes_infos_temporal_train.pkl",
    "nuscenes_infos_temporal_val.pkl",
]

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/nuscenes")
    args = parser.parse_args()
    root = Path(args.data_root)
    mini = root / "v1.0-mini"
    missing = [str(mini / name) for name in REQUIRED_JSON if not (mini / name).exists()]
    missing += [str(root / "samples" / cam) for cam in CAMERAS if not (root / "samples" / cam).exists()]
    missing += [
        str(root / name)
        for name in REQUIRED_METADATA
        if not (root / name).is_file() or (root / name).stat().st_size == 0
    ]
    can_bus = root.parent / "can_bus"
    if not can_bus.is_dir() or not any(can_bus.glob("*.json")):
        missing.append(str(can_bus))
    if missing:
        print("Missing nuScenes mini files/directories:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)
    camera_files = sum(
        1
        for camera in CAMERAS
        for path in (root / "samples" / camera).iterdir()
        if path.is_file()
    )
    print(
        f"nuScenes mini layout is ready under {root}; "
        f"camera files={camera_files}, temporal metadata=2, CAN bus present"
    )


if __name__ == "__main__":
    main()
