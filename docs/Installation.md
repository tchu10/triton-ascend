# 安装指南
# Python wheel安装
通过 Python Wheel 安装包进行安装是最快捷、最简便的方式。使用下面命令安装：
```
pip install triton-ascend==3.2.0rc1
```
# 源代码安装
## 前置步骤

### Python版本要求

当前Triton-Ascend要求的Python版本为:**py3.9-py3.11**。

### 安装Ascend CANN
异构计算架构CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，
向上支持多种AI框架，包括MindSpore、PyTorch、TensorFlow等，向下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平
台。

您可以访问昇腾社区官网，根据其提供的软件安装指引完成 CANN 的安装配置。

在安装过程中，请选择 CANN 版本 8.2.RC1.alpha002，并根据实际环境指定操作系统、安装方式和业务场景。

社区下载链接：
```
https://www.hiascend.com/developer/download/community/result?module=cann
```
社区安装指引链接：
```
https://www.hiascend.com/developer/download/community/result?module=cann
```
该文档提供了完整的安装流程说明与依赖项配置建议，适用于需要全面部署 CANN 环境的用户。

### 安装python依赖
```
pip install decorator cffi protobuf==3.20 attrs pyyaml pathlib2 scipy psutil absl-py tvm cloudpickle pybind11 einops pytest te numpy
```

### 安装torch_npu

当前配套的torch_npu版本为2.6.0rc1版本。
```
pip install torch_npu==2.6.0rc1
```

## 源码编译安装

### **系统要求**

- GCC >= 9.4.0
- GLIBC >= 2.29
- clang

### 安装系统库依赖

安装zlib1g-dev/lld/clang，可选安装ccache包用于加速构建。
- 推荐版本 clang >= 15
- 推荐版本 lld >= 15
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

### **克隆 Triton-Ascend**

```
git clone https://gitee.com/ascend/triton-ascend.git --recurse-submodules --shallow-submodules
```

### **基于LLVM构建**

Triton 使用 LLVM20 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。

1. `git checkout` 检出指定版本的LLVM.

   ```
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
   ```

2. 构建LLVM。可以运行以下命令：
- 注：请在下面指令中设置您想安装LLVM的目标路径 -DCMAKE_INSTALL_PREFIX=yourpath/llvm-install

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
   ninja install
   ```
- 说明：若环境上ccache已安装且正常运行，可设置`-DLLVM_CCACHE_BUILD=ON`加速构建, 否则请勿开启。
- clang安装LLVM
  
  可使用clang安装LLVM，环境上按安装clang、lld，并指定版本(推荐版本clang>=15，lld>=15)，
  以下面指令安装clang，：
  ``` 
  apt-get install -y clang-15 lld-15 ccache
  ``` 
  如果环境上有多个版本的clang，请设置clang为当前安装的版本clang-15:
  ``` 
  update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20; \
  update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20; \
  update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 20
  ```
  设置C编译器为clang，以下面指令安装LLVM：
  ```
  cd build
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DCMAKE_INSTALL_PREFIX=yourpath/llvm-install \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON
  ninja install
  ```
### **构建 Triton-Ascend**

1. 源码安装

- 注1：请在下面指令中设置您在上一步LLVM安装的目标路径 LLVM_SYSPATH=yourpath/llvm-install
- 注2：请确保已安装clang>=15，lld>=15，TRITON_BUILD_WITH_CLANG_LLD=true使用了clang和lld
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
   如果已安装`ccache`，可以使用以下命令加速编译 TRITON_BUILD_WITH_CCACHE=true。
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
   # 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 运行tutorials示例：
   python3 ./triton-ascend/ascend/examples/tutorials/01-vector-add.py
   ```
