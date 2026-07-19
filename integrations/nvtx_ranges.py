from contextlib import contextmanager


@contextmanager
def nvtx_range(name):
    pushed = False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
    except Exception:
        pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                import torch

                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
