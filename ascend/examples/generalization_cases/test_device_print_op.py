import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common


def torch_pointwise(x0, x1):
    res = x0 - x1
    return res

@triton.jit
def triton_sub(in_ptr0, in_ptr1, out_ptr0, DUMMY0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 - tmp1
        if (loop1 == 0):
            tl.device_print("DUMMY0 = ", DUMMY0)
            tl.device_print("OUTPUT = ", tmp2)
        tl.store(out_ptr0 + (x0), tmp2, None)

@pytest.mark.parametrize('param_list',
                         [
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                             ['int32', (2, 4096, 8), 2, 32768, 1024],
                             ['int16', (2, 4096, 8), 2, 32768, 1024],
                             ['int64', (2, 4096, 8), 2, 32768, 1024],
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                         ]
                         )
def test_case(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch_pointwise(x0, x1)
    y_cal = torch.zeros(shape, dtype = eval('torch.' + dtype)).npu()
    dummys = torch.randn((16,), dtype = torch.float32)
    triton_sub[ncore, 1, 1](x0, x1, y_cal, dummys[0].item(), xblock, xblock_sub, debug=True)
    test_common.validate_cmp(dtype, y_cal, y_ref)