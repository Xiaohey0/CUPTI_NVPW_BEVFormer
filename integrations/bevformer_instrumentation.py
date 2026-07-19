import importlib
import json
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from types import MethodType

from bevformer_stage_hooks import bev_stage, current_stage


STAGE_CLASS_MAP = {
    "TemporalSelfAttention": "temporal_self_attention",
    "SpatialCrossAttention": "spatial_cross_attention",
    "MSDeformableAttention3D": "ms_deformable_attention",
}

_STAGE_CALLS = OrderedDict()


def import_bevformer_plugins(cfg):
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if getattr(cfg, "plugin", False):
        if hasattr(cfg, "plugin_dir"):
            module_path = cfg.plugin_dir.replace("/", ".").strip(".")
        else:
            module_path = "projects.mmdet3d_plugin"
        importlib.import_module(module_path)


def _stage_for_module(name, module):
    if name == "img_backbone":
        return "image_backbone"
    if name == "img_neck":
        return "neck_fpn"
    if name == "pts_bbox_head":
        return "detection_head"
    return STAGE_CLASS_MAP.get(module.__class__.__name__)


def _shape_summary(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(x) for x in shape]
    if isinstance(value, (list, tuple)):
        return [
            [int(x) for x in item.shape]
            for item in value
            if getattr(item, "shape", None) is not None
        ]
    return None


def _shape_config_from_call(args, kwargs):
    names = [
        "query",
        "key",
        "value",
        "reference_points",
        "spatial_shapes",
        "level_start_index",
        "mlvl_feats",
        "img",
    ]
    values = dict(zip(names, args))
    values.update({key: value for key, value in kwargs.items() if key in names})
    summary = {}
    for key, value in values.items():
        shape = _shape_summary(value)
        if shape:
            summary[key] = shape
    return summary


def _record_stage_call(stage_name, module_name, class_name, args, kwargs):
    shape_config = _shape_config_from_call(args, kwargs)
    key = json.dumps(
        {
            "stage_name": stage_name,
            "module_name": module_name,
            "class_name": class_name,
            "shape_config": shape_config,
        },
        sort_keys=True,
    )
    if key not in _STAGE_CALLS:
        _STAGE_CALLS[key] = {
            "stage_name": stage_name,
            "module_name": module_name,
            "class_name": class_name,
            "shape_config": shape_config,
            "calls": 0,
        }
    _STAGE_CALLS[key]["calls"] += 1


def install_module_stage_hooks(model):
    wrapped = []
    for name, module in model.named_modules():
        stage_name = _stage_for_module(name, module)
        if stage_name is None or getattr(module, "_bev_profiler_wrapped", False):
            continue
        original_forward = module.forward
        class_name = module.__class__.__name__

        @wraps(original_forward)
        def wrapped_forward(
            self,
            *args,
            __orig=original_forward,
            __stage=stage_name,
            __name=name,
            __class_name=class_name,
            **kwargs,
        ):
            _record_stage_call(
                __stage, __name, __class_name, args, kwargs
            )
            with bev_stage(__stage):
                return __orig(*args, **kwargs)

        module.forward = MethodType(wrapped_forward, module)
        module._bev_profiler_wrapped = True
        wrapped.append((name, class_name, stage_name))

    head = getattr(model, "pts_bbox_head", None)
    if head is not None and hasattr(head, "get_bboxes"):
        original_get_bboxes = head.get_bboxes

        @wraps(original_get_bboxes)
        def wrapped_get_bboxes(self, *args, **kwargs):
            _record_stage_call(
                "bbox_decode",
                "pts_bbox_head.get_bboxes",
                self.__class__.__name__,
                args,
                kwargs,
            )
            with bev_stage("bbox_decode"):
                return original_get_bboxes(*args, **kwargs)

        head.get_bboxes = MethodType(wrapped_get_bboxes, head)
        wrapped.append(
            ("pts_bbox_head.get_bboxes", head.__class__.__name__, "bbox_decode")
        )
    return wrapped


def write_stage_shape_manifest(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(list(_STAGE_CALLS.values()), indent=2) + "\n",
        encoding="utf-8",
    )


def install_msda_tensor_capture(
    output_path, target_stage="ms_deformable_attention", overwrite=False
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    func_mod = importlib.import_module(
        "projects.mmdet3d_plugin.bevformer.modules."
        "multi_scale_deformable_attn_function"
    )
    state = {"call_index": 0}

    def patch_function(cls):
        if getattr(cls, "_bev_capture_wrapped", False):
            return
        original_forward = cls.forward

        @staticmethod
        @wraps(original_forward)
        def wrapped_forward(
            ctx,
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        ):
            state["call_index"] += 1
            stage_name = current_stage()
            if not output_path.exists() and stage_name == target_stage:
                import torch

                payload = {
                    "capture_source": "real_bevformer_forward",
                    "capture_stage": stage_name,
                    "capture_call_index": state["call_index"],
                    "capture_function": cls.__name__,
                    "value": value.detach().contiguous(),
                    "spatial_shapes": value_spatial_shapes.detach().contiguous(),
                    "level_start_index": (
                        value_level_start_index.detach().contiguous()
                    ),
                    "sampling_locations": (
                        sampling_locations.detach().contiguous()
                    ),
                    "attention_weights": (
                        attention_weights.detach().contiguous()
                    ),
                    "im2col_step": int(im2col_step),
                }
                temporary = output_path.with_suffix(output_path.suffix + ".tmp")
                torch.save(payload, temporary)
                temporary.replace(output_path)
            return original_forward(
                ctx,
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            )

        cls.forward = wrapped_forward
        cls._bev_capture_wrapped = True

    patch_function(func_mod.MultiScaleDeformableAttnFunction_fp32)
    patch_function(func_mod.MultiScaleDeformableAttnFunction_fp16)
    return True
