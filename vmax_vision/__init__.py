"""VMAX steward vision pipeline.

An independent detection / tracking / boundary-judgement stack that consumes the
rendered clips in ``output/`` as ordinary video. Ground truth is used only to
train the detector on the held-in split and to score results afterwards; nothing
in :mod:`vmax_vision.pipeline` reads a per-frame label at inference time.
"""
__version__ = "1.0.0"
