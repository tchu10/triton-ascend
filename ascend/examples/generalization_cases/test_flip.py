# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
import logging
from test_common import TestUtils, check_ub_mem_overflow
import triton.language.extra.ascend.libdevice as libdevice

@triton.jit
def fn_npu_1d(output_ptr, x_ptr, XB: tl.constexpr):
    xidx = tl.arange(0, XB)
    idx = xidx
    X = tl.load(x_ptr + idx)
    ret = libdevice.flip(X, 0)
    oidx = xidx
    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_2d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB)
    idx = xidx[:, None] * YB + yidx[None, :]
    X = tl.load(x_ptr + idx)
    ret = libdevice.flip(X, 1)
    oidx = xidx[:, None] * YB + yidx[None, :]
    tl.store(output_ptr + oidx, ret)

@triton.jit
def fn_npu_3d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    ret = libdevice.flip(X, 2)
    oidx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]
    tl.store(output_ptr + oidx, ret)

typelist = ['int8','int16','int32','int64','float16','bfloat16','float32', 'bool']
#typelist = ['int64', 'bool', 'bfloat16'] # error dtypes

dtype_mapping = {
    'int8': (torch.int8),
    'int16': (torch.int16),
    'int32': (torch.int32),
    'uint32': (torch.uint32),
    'int64': (torch.int64),
    'float16': (torch.float16),
    'float32': (torch.float32),
    'bfloat16': (torch.bfloat16),
    'bool': (torch.bool),
}

@pytest.mark.parametrize('shape', TestUtils.test_shape1_2_3d)
@pytest.mark.parametrize('dtype',typelist)
def test_flip(shape, dtype):
    logging.debug(f'dtype:{dtype} shape:{shape}')
    data_dtype = eval('torch.' + dtype)
    x = None
    if dtype == 'bool':
        x = torch.randint(low=0, high=2, size=shape, dtype=data_dtype).npu()
    else:
        x = torch.randint(low=0, high=128, size=shape, dtype=data_dtype).npu()

    # torch_npu䷾M弾T¯弾L~Auint32漾Z~Dflip
    torch_input = x if x.dtype != torch.uint32 else x.to(torch.float32)
    torch_res = torch.flip(torch_input, dims=(-1,))
    triton_res = torch.empty(shape, dtype=data_dtype).npu()
    if len(shape) == 1:
        fn_npu_1d[1, 1, 1](triton_res, x, shape[0])
    elif len(shape) == 2:
        shape0 = shape[0]
        shape1 = shape[1]
        if x.numel() * x.element_size() >= 8192:
            grid = (shape0, 1, 1)
            shape0 = 1
        else:
            grid = (1, 1, 1)
        fn_npu_2d[grid](triton_res, x, shape0, shape1)
    elif len(shape) == 3:
        fn_npu_3d[1, 1, 1](triton_res, x, shape[0], shape[1], shape[2])

    triton_res = triton_res if triton_res.dtype != torch.uint32 else triton_res.to(torch.float32)
    cmp_dtype = dtype if dtype != 'uint32' else 'float32'
    test_common.validate_cmp(cmp_dtype, triton_res, torch_res)


if __name__ == "__main__":
    for dtype in TestUtils.dtype_list:
        for shape in [(37, 3), (1, 22, 39)]:
            test_flip(shape, dtype)
