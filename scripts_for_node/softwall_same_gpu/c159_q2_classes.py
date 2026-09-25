#!/usr/bin/env python3.11
"""Frozen Qwen context classes and physical-completion bounds for C159-Q2."""

CONTEXT_LENGTHS = (16, 32, 64, 128, 256, 512)
CLASS_BOUNDS_MS = {16: 35, 32: 35, 64: 35, 128: 40, 256: 65, 512: 75}

