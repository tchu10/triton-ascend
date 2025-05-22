# Triton Ascend

Triton是一种编程语言和编译器，用于高效编写定制的深度学习原语。其目标是提供一个开源环境，让开发者能够高效开发代码，同时兼具比其他现有领域专用语言DSL（domain-specific language）更强的灵活性。

Triton-Ascend面向昇腾平台，旨在让Triton代码能够在昇腾硬件上高效运行。

# 源码安装

## **系统要求**

- GCC >= 9.4.0
- GLIBC >= 2.29
- clang

## **依赖**

### 包版本依赖

Python支持版本为:**py3.9-py3.11**, torch及torch_npu支持版本为:**2.6.0**。

### 安装系统库依赖

安装zlib1g-dev/lld/clang，可选安装ccache包用于加速构建。
```
以ubuntu系统为例：
apt update
apt install zlib1g-dev lld clang
apt install ccache # optional
```

### 安装python依赖
```
pip install ninja cmake wheel pybind11 # build-time dependencies
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml torch==2.6.0 torch-npu==2.6.0rc1 # torch dependencies
```

## **克隆 Triton-Ascend**

```
git clone --recurse-submodules https://gitee.com/ascend/triton-ascend.git
```

## **基于LLVM构建**

Triton 使用 LLVM20 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。

1. `git checkout` 检出指定版本的LLVM.

   ```
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
   ```

2. 构建LLVM。可以运行以下命令：

   ```
   cd $HOME/llvm-project  # your clone of LLVM.
   mkdir build
   cd build
   cmake -G Ninja  ../llvm  \
      -DLLVM_CCACHE_BUILD=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm"  \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
      -DCMAKE_INSTALL_PREFIX=yourpath/llvm-install
   ninja -j 32 install
   ```
   - 说明：若环境上ccache已安装且正常运行，可设置`-DLLVM_CCACHE_BUILD=ON`加速构建, 否则请勿开启。

## **构建 Triton-Ascend**

1. 源码安装
   ```
   cd triton-ascend/
   LLVM_SYSPATH=yourpath/llvm-install \
   TRITON_PLUGIN_DIRS=./ascend \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```
   如果已安装`ccache`，可以使用以下命令加速编译：
   ```
   cd triton-ascend/
   LLVM_SYSPATH=yourpath/llvm-install \
   TRITON_PLUGIN_DIRS=./ascend \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```

2. 运行Triton示例
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   pytest ./ascend/examples/pytest_ut/test_add.py
   ```


## **环境变量**

- `MLIR_ENABLE_DUMP=1`： 在每次 MLIR 优化前转储所有内核的 IR。使用 `MLIR_ENABLE_DUMP=kernelName`可以只转储特定内核的IR。Triton 缓存可能干扰转储。如果 `MLIR_ENABLE_DUMP=1`  不生效，可尝试清理 Triton 缓存： `rm -r ~/.triton/cache/*`。

- `TRITON_INTERPRET=1`： 使用 Triton 解释器而非 NPU 运行，支持在核函数代码中插入 Python 断点。

- `TRITON_PRINT_AUTOTUNING=1`：在自动调优完成后，输出每个内核的最佳配置及总耗时。

- `TRITON_ALWAYS_COMPILE=1`：强制重新编译内核（忽略缓存命中）。

- `MLIR_ENABLE_TIMING`：转储各MLIR优化阶段的耗时信息。

- `TRITON_KERNEL_DUMP`：启用各编译阶段IR的转储功能。

- `TRITON_DUMP_DIR`：指定当`TRITON_KERNEL_DUMP`时保存IR的目录。

- `TRITON_DEVICE_PRINT`：当设置为`1` 或者 `true`时（`TRUE` 将被转换为 `true`），启用`tl.device_print`功能。当前每个线程的GM缓冲区最大为16KB，超限内容将被丢弃。该值目前固定，后续将通过环境变量调整。

- `TRITON_BENCH_METHOD=npu`：使用昇腾NPU时，将`testing.py`中的`do_bench`切换为`do_bench_npu`（需配合`INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE=1`使用）。设为`default`时即使NPU可用，仍调用原函数。

- `TRITON_ASCEND_COMPILE_SPEED_OPT`：控制JIT编译器在发现内核编译失败后是否跳过后续编译阶段。设为`1`跳过（默认`0`继续尝试）。

- `TRITON_ENABLE_SANITIZER`: 开启后，才能配合mssanitize工具进行有效检测。默认关闭。

## 示例

环境配置完成后，可通过教程脚本快速上手，教程路径：`triton-ascend/ascend/examples/tutorials`

```
cd triton-ascend/ascend/examples/tutorials
# take 01-vector-add.py for example
python3 01-vector-add.py
```

## 调试Triton-Ascend

参考triton社区提供的调试方法进行调试，官方链接：https://triton-lang.org/main/programming-guide/chapter-3/debugging.html

## 当前支持的Ascend设备

  - 已支持：Atlas 800T/I A2产品
  - 开发中：Atlas 800T/I A3产品

## 当前支持的triton op列表

|        Triton Op Type    |        Triton Op       |
|:------------------------:|:----------------------:|
|       Creation Ops       | arange                 |
|                          | cat                    |
|                          | full                   |
|                          | zeros                  |
|                          | zeros_like             |
|                          | cast                   |
|  Shape Manipulation Ops  | broadcast              |
|                          | broadcast_to           |
|                          | expand_dims            |
|                          | interleave             |
|                          | join                   |
|                          | permute                |
|                          | ravel                  |
|                          | reshape                |
|                          | split                  |
|                          | trans                  |
|                          | view                   |
|    Linear Algebra Ops    | dot                    |
|    Memory/Pointer Ops    | load                   |
|                          | store                  |
|                          | make_block_ptr         |
|                          | advance                |
|       Indexing Ops       | flip                   |
|                          | where                  |
|                          | swizzle2d              |
|         Math Ops         | add                    |
|                          | sub                    |
|                          | mul                    |
|                          | div                    |
|                          | floordiv(//)           |
|                          | mod                    |
|                          | neg                    |
|                          | invert(!)              |
|                          | and(&)                 |
|                          | or(\|)                 |
|                          | xor(^)                 |
|                          | not(~)                 |
|                          | lshift(<<)             |
|                          | rshift(>>)             |
|                          | gt                     |
|                          | ge                     |
|                          | lt                     |
|                          | le                     |
|                          | eq                     |
|                          | ne                     |
|                          | logical and            |
|                          | logical or             |
|                          | abs                    |
|                          | cdiv                   |
|                          | ceil                   |
|                          | clamp                  |
|                          | cos                    |
|                          | div_rn                 |
|                          | erf                    |
|                          | exp                    |
|                          | exp2                   |
|                          | fdiv                   |
|                          | floor                  |
|                          | fma                    |
|                          | log                    |
|                          | log2                   |
|                          | maximum                |
|                          | minimum                |
|                          | rsqrt                  |
|                          | sigmoid                |
|                          | sin                    |
|                          | softmax                |
|                          | sqrt                   |
|                          | sqrt_rn                |
|                          | umulhi                 |
|       Reduction Ops      | argmax                 |
|                          | argmin                 |
|                          | max                    |
|                          | min                    |
|                          | reduce                 |
|                          | sum                    |
|                          | xor_sum                |
|      Scan/Sort Ops       | gather                 |
|        Atomic Ops        | atomic_add             |
|                          | atomic_max             |
|                          | atomic_min             |
| Random Number Generation | randint4x              |
|                          | randint                |
|                          | rand                   |
|                          | randn                  |
|         Iterators        | range                  |
|                          | static_range           |
|     Compiler Hint Ops    | debug_barrier          |
|         Debug Ops        | static_print           |
|                          | static_assert          |
|                          | device_print           |

各op支持度及使用约束参见[Python_API.md](./docs/Python_API.md)

## 当前支持的开源算子仓算子列表

### FlagGems:
- abs
- add
- bitwise_and
- bitwise_not
- bitwise_or
- cos
- div
- eq
- exp
- ge
- gt
- isinf
- rsub
- le
- lt
- mul
- ne
- neg
- reciprocal
- relu
- rsqrt
- sigmoid
- silu
- sin
- sub
- tanh
- triu

其他开源算子仓（如vllm、sglang等）正在逐步支持中，敬请期待。
