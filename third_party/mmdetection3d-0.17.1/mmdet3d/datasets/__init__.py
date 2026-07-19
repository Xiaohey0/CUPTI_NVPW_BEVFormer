# Copyright (c) OpenMMLab. All rights reserved.
"""Minimal real dataset registry for the camera-only BEVFormer experiment."""

from mmdet.datasets.builder import build_dataloader

from .builder import DATASETS, build_dataset
from .custom_3d import Custom3DDataset
from .nuscenes_dataset import NuScenesDataset
from .pipelines import (Compose, DefaultFormatBundle3D,
                        LoadMultiViewImageFromFiles, MultiScaleFlipAug3D)

__all__ = [
    'build_dataloader', 'DATASETS', 'build_dataset', 'Custom3DDataset',
    'NuScenesDataset', 'Compose', 'DefaultFormatBundle3D',
    'LoadMultiViewImageFromFiles', 'MultiScaleFlipAug3D'
]
