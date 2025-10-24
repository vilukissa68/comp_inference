#!/usr/bin/env python3

"""Data type enumeration and mapping for compressed model parameters."""
import torch
from enum import IntEnum


class DTypeEnum(IntEnum):
    """Enumeration for supported torch data types."""

    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2
    INT32 = 3
    INT64 = 4


DTYPE_TO_ENUM = {
    torch.float32: DTypeEnum.FLOAT32,
    torch.float16: DTypeEnum.FLOAT16,
    torch.bfloat16: DTypeEnum.BFLOAT16,
    torch.int32: DTypeEnum.INT32,
    torch.int64: DTypeEnum.INT64,
}

ENUM_TO_DTYPE = {v: k for k, v in DTYPE_TO_ENUM.items()}
