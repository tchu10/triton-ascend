# PYTHON API
## 1. triton op支持度总表

|                          |        Triton Op       | int8 | int16 | int32 | uint32 | int64 | fp16 | fp32 | bf16 | bool |
|:------------------------:|:----------------------:|------|-------|-------|--------|-------|------|------|------|------|
|       Creation Ops       | arange                 | ✓    | ✓     | ✓     | ×      | ×     | ×    | ×    | ×    | NA   |
|                          | cat                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | full                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | zeros                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | zeros_like             | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | cast                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|  Shape Manipulation Ops  | broadcast              | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | broadcast_to           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | expand_dims            | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | interleave             | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | join                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | permute                | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | NA   |
|                          | ravel                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | reshape                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | split                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | trans                  | ✓    | ✓     | ✓     | ×      | ×     | ✓    | ✓    | ✓    | NA   |
|                          | view                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|    Linear Algebra Ops    | dot                    | ✓    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | dot_scaled             | NA   | NA    | NA    | NA     | NA    | NA   | NA   | NA   | NA   |
|    Memory/Pointer Ops    | load                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | store                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | make_block_ptr         | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | advance                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|       Indexing Ops       | flip                   | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | where                  | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | swizzle2d              | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|         Math Ops         | add                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | sub                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | mul                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | div                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | floordiv(//)           | ×    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | mod                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | neg                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | invert(!)              | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | and(&)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | or(\|)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | xor(^)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | not(~)                 | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ✓    |
|                          | lshift(<<)             | ✓    | ✓     | ✓     | ×      | ✓     | NA   | NA   | NA   | NA   |
|                          | rshift(>>)             | ✓    | ✓     | ✓     | ×      | ✓     | NA   | NA   | NA   | NA   |
|                          | gt                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | ge                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | lt                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | le                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | eq                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | ne                     | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | logical and            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ✓    |
|                          | logical or             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ✓    |
|                          | abs                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | cdiv                   | ×    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | ×    |
|                          | ceil                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | clamp                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | cos                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | div_rn                 | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | erf                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | exp                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | exp2                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | fdiv                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | floor                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | fma                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | log                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | log2                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | maximum                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | minimum                | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | NA   |
|                          | rsqrt                  | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sigmoid                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sin                    | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | softmax                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sqrt                   | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | sqrt_rn                | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|                          | umulhi                 | ×    | ×     | ✓     | ×      | ×     | ×    | ×    | ×    | ×    |
|       Reduction Ops      | argmax                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | argmin                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ×    |
|                          | max                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | NA   | ×    |
|                          | min                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | NA   | ×    |
|                          | reduce                 | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | NA   | ×    |
|                          | sum                    | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | NA   | NA   |
|                          | xor_sum                | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | NA   |
|       Scan/Sort Ops      | associative_scan       | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | cumprod                | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | cumsum                 | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | histogram              | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | sort                   | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | gather                 | ×    | ×     | ×     | ×      | ×     | ✓    | ✓    | ✓    | ×    |
|        Atomic Ops        | atomic_add             | ×    | ×     | ✓     | ×      | ✓     | ✓    | ✓    | ×    | ×    |
|                          | atomic_and             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_cas             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_max             | ×    | ×     | ✓     | ×      | ✓     | ×    | ✓    | ×    | ×    |
|                          | atomic_min             | ×    | ×     | ✓     | ×      | ✓     | ×    | ✓    | ×    | ×    |
|                          | atomic_or              | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_xchg            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | atomic_xor             | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
| Random Number Generation | randint4x              | NA   | NA    | ✓     | ×      | NA    | NA   | ✓    | NA   | NA   |
|                          | randint                | NA   | NA    | ✓     | ×      | NA    | NA   | ✓    | NA   | NA   |
|                          | rand                   | NA   | NA    | ✓     | ×      | NA    | NA   | ✓    | NA   | NA   |
|                          | randn                  | NA   | NA    | ✓     | ×      | NA    | NA   | ✓    | NA   | NA   |
|         Iterators        | range                  | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | NA   |
|                          | static_range           | ✓    | ✓     | ✓     | ×      | ✓     | ×    | ×    | ×    | NA   |
|      Inline Assembly     | inline_asm_elementwise | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|     Compiler Hint Ops    | debug_barrier          | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | max_constancy          | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | max_contiguous         | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|                          | multiple_of            | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |
|         Debug Ops        | static_print           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | static_assert          | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ✓    | ✓    |
|                          | device_print           | ✓    | ✓     | ✓     | ×      | ✓     | ✓    | ✓    | ×    | ✓    |
|                          | device_assert          | ×    | ×     | ×     | ×      | ×     | ×    | ×    | ×    | ×    |

### 约束说明
- dot: 两个输入A[batch(optional), M, K], B[batch(optional), K, N]，M，N按照16对齐，K按照32B对齐。

- gather: triton.gather(x, index, axis)，假设x的shape为n维度，目前只支持axis=n-1。

- permute: triton.permute(x, dims)，不支持dims=[2, 1, 0]。

- trans: triton.trans(x, dims)，不支持dims=[2, 1 , 0]。

- device_print: 需要增加2个环境变量，TRITON_DEVICE_PRINT=1，TRITON_ENABLE_TASKQUEUE=0。TRITON_ENABLE_TASKQUEUE=0可能造成程序运行不稳定，建议仅临时关闭。

- atomic_add: 不支持标量（包括长度为1的tensor）访存

- atomic_max: 不支持标量（包括长度为1的tensor）访存

- atomic_min: 不支持标量（包括长度为1的tensor）访存

## 2. Testing
### 基础用例
各类算子的基础用例，代码路径如下，可单个用例执行，也可多线程执行所有用例
```
cd /triton-ascend/ascend/examples/pytest_ut

# 单个用例执行：test_add.py 示例
pytest test_add.py

# 多线程执行所有用例：-n 16，以16个线程执行当前目录下所有用例
pytest ./ -n 16
```
