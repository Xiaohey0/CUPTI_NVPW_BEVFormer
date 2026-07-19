# Copyright (c) OpenMMLab. All rights reserved.
"""Pipeline components required by BEVFormer tiny inference."""

from mmdet.datasets.pipelines import Compose

from .formating import DefaultFormatBundle3D
from .loading import LoadMultiViewImageFromFiles
from .test_time_aug import MultiScaleFlipAug3D

__all__ = [
    'Compose', 'DefaultFormatBundle3D', 'LoadMultiViewImageFromFiles',
    'MultiScaleFlipAug3D'
]
