import ctypes
import os
from contextlib import ContextDecorator
from contextvars import ContextVar

from nvtx_ranges import nvtx_range


_STAGE_STACK = ContextVar("bevformer_stage_stack", default=())


def current_stage():
    stack = _STAGE_STACK.get()
    return stack[-1] if stack else ""


def _load_library(env_name, default_name):
    configured = os.environ.get(env_name)
    candidates = [configured] if configured else []
    candidates.append(default_name)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


class ProfilerBridge:
    def __init__(self):
        self.activity = _load_library(
            "BEV_ACTIVITY_LIB", "libbevformer_activity_profiler.so"
        )
        if self.activity:
            self.activity.bev_profiler_begin_request.argtypes = [ctypes.c_int]
            self.activity.bev_profiler_push_stage.argtypes = [ctypes.c_char_p]

    def begin_request(self, request_id):
        if self.activity:
            self.activity.bev_profiler_begin_request(int(request_id))

    def end_request(self):
        if self.activity:
            self.activity.bev_profiler_end_request()

    def push_stage(self, stage_name):
        if self.activity:
            self.activity.bev_profiler_push_stage(stage_name.encode("utf-8"))

    def pop_stage(self):
        if self.activity:
            self.activity.bev_profiler_pop_stage()


BRIDGE = ProfilerBridge()


class bev_stage(ContextDecorator):
    def __init__(self, stage_name, request_id=None):
        self.stage_name = stage_name
        self.request_id = request_id
        self._started_request = False
        self._stack_token = None
        self._nvtx = None

    def __enter__(self):
        if self.request_id is not None:
            BRIDGE.begin_request(self.request_id)
            self._started_request = True
        stack = _STAGE_STACK.get()
        self._stack_token = _STAGE_STACK.set(stack + (self.stage_name,))
        BRIDGE.push_stage(self.stage_name)
        self._nvtx = nvtx_range(self.stage_name)
        self._nvtx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._nvtx:
            self._nvtx.__exit__(exc_type, exc, tb)
        BRIDGE.pop_stage()
        if self._stack_token is not None:
            _STAGE_STACK.reset(self._stack_token)
        if self._started_request:
            BRIDGE.end_request()
        return False
