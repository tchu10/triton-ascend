#!/bin/bash
set -ex

WORK_ROOT=$(pwd)
TRITON_VERSION=${1}

SHARED_DIR=/home/shared
mkdir -p $SHARED_DIR

declare -A py_map=(["py39"]="3.9" ["py310"]="3.10" ["py311"]="3.11")

# TODO: check this file exists
source /opt/miniconda3/etc/profile.d/conda.sh

for env_name in "${!py_map[@]}"; do
    if ! conda env list | grep -wq "$env_name"; then
      conda create -y -n "$env_name" python="${py_map[$env_name]}"
    fi
    conda activate $env_name
    pip3 install pybind11 -i https://repo.huaweicloud.com/repository/pypi/simple/
done

# LLVM
# TODO: upload to gitee or huawei-yellow intranet. Then we only need to download it.
cd $SHARED_DIR
if [ ! -d llvm-project ]; then
  git clone --depth 1 https://gitclone.com/github.com/llvm/llvm-project.git
fi
cd llvm-project
TARGET_COMMIT="b5cc222d7429fe6f18c787f633d5262fac2e676f"
if ! git rev-parse --quiet --verify "$TARGET_COMMIT" &>/dev/null; then
  echo "Error: Commit $TARGET_COMMIT does not exist!" >&2
  exit 1
fi
if [ "$(git rev-parse HEAD)" != "$TARGET_COMMIT" ]; then
  echo "LLVM commit is NOT $TARGET_COMMIT"
  git fetch --depth 1 origin b5cc222d7429fe6f18c787f633d5262fac2e676f
  git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
else
  echo "LLVM commit is $TARGET_COMMIT"
fi
if [ ! -d /opt/llvm-b5cc222/lib ]; then
  mkdir -p build
  cd build
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm-b5cc222 \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON
  ninja install
fi
# triton-ascend
cd $WORKSPACE

triton_commit=$(git submodule status | awk '{print $1}' | cut -c 2-)
triton_url=$(git config --file=.gitmodules submodule.triton.url)
mirror_triton_url=$(echo "$triton_url" | sed 's|https://github.com|https://gitclone.com/github.com|')
git clone --depth 1 ${mirror_triton_url}
cd triton
git fetch --depth 1 origin ${triton_commit}
git checkout ${triton_commit}
# In the future, we will propose a PR to apply the following change.

cd $WORKSPACE

environments=("py39" "py310" "py311")

for env in "${environments[@]}"; do
    (
        conda activate "$env" && \
        echo "在 $env 中执行命令" && \
        bash build/build_triton_ascend.sh $(pwd)/triton $(pwd)/ascend /opt/llvm-b5cc222 ${TRITON_VERSION} bdist_wheel
    ) || echo "$env 执行失败"
done
