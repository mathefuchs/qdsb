from __future__ import annotations


class RunningMean:
    """Minimal checkpoint-compatibility shim for legacy ALAE trackers."""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __getstate__(self):
        return dict(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)


class RunningMeanTorch(RunningMean):
    pass
