#!/bin/bash
set -ex

COMPILER_ROOT=/home/shared/bisheng_toolkit_20250519

function build_and_test() {
  if [ -d ${HOME}/.triton/dump ];then
    rm -rf ${HOME}/.triton/dump
  fi
  if [ -d ${HOME}/.triton/cache ];then
    rm -rf ${HOME}/.triton/cache
  fi

  if [ -d "${WORKSPACE}/triton" ];then
    rm -rf "${WORKSPACE}/triton"
  fi
  cd ${WORKSPACE}
  triton_commit=$(git submodule status | awk '{print $1}' | cut -c 2-)
  triton_url=$(git config --file=.gitmodules submodule.triton.url)
  mirror_triton_url=$(echo "$triton_url" | sed 's|github.com/triton-lang|gitee.com/shijingchang|')
  git clone --depth 1 ${mirror_triton_url}
  cd triton
  git fetch --depth 1 origin ${triton_commit}
  git checkout ${triton_commit}

  bash ${WORKSPACE}/build/build_triton_ascend.sh \
    ${WORKSPACE}/triton \
    ${WORKSPACE}/ascend \
    ${LLVM_BUILD_DIR} \
    1.0 \
    install

  if [ -d __pycache__ ];then
    rm -rf __pycache__
  fi
  if [ -d ${HOME}/.triton/cache ]; then
    rm -rf ${HOME}/.triton/cache
  fi
  cd ${WORKSPACE}/ascend/examples/pytest_ut
  pytest -n 16 --dist=load . || { exit 1 ; }
}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LLVM_BUILD_DIR=/opt/llvm-b5cc222
# FIXME: 20250508 the bishengir-compile in the CANN 8.0.T115 fails lots of cases
#        So we need to use another version of compiler.
export PATH=${COMPILER_ROOT}:${COMPILER_ROOT}/ccec_compiler/bin:$PATH

# build in torch 2.3.1
source /opt/miniconda3/bin/activate torch_231
build_and_test

# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_and_test
