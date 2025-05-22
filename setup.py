import os
import platform
import re
import contextlib
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
import urllib.request
import json
from io import BytesIO
from distutils.command.clean import clean
from pathlib import Path
from typing import List, NamedTuple, Optional

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from dataclasses import dataclass

from distutils.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from wheel.bdist_wheel import bdist_wheel

import pybind11

script_dir = os.path.dirname(__file__)
triton_python_dir = os.path.join(os.path.dirname(__file__), "triton/python")

@dataclass
class Backend:
    name: str
    package_data: List[str]
    language_package_data: List[str]
    src_dir: str
    backend_dir: str
    language_dir: Optional[str]
    install_dir: str
    is_external: bool


class BackendInstaller:

    @staticmethod
    def prepare(
        backend_name: str, backend_src_dir: str = None, is_external: bool = False
    ):
        # Initialize submodule if there is one for in-tree backends.
        if not is_external:
            root_dir = os.path.join(os.pardir, "third_party")
            assert backend_name in os.listdir(
                root_dir
            ), f"{backend_name} is requested for install but not present in {root_dir}"

            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", f"{backend_name}"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    cwd=root_dir,
                )
            except subprocess.CalledProcessError:
                pass
            except FileNotFoundError:
                pass

            backend_src_dir = os.path.join(root_dir, backend_name)

        backend_path = os.path.abspath(os.path.join(backend_src_dir, "backend"))
        assert os.path.exists(backend_path), f"{backend_path} does not exist!"

        language_dir = os.path.abspath(os.path.join(backend_src_dir, "language"))
        if not os.path.exists(language_dir):
            language_dir = None

        for file in ["compiler.py", "driver.py"]:
            assert os.path.exists(
                os.path.join(backend_path, file)
            ), f"${file} does not exist in ${backend_path}"

        install_dir = os.path.join(
            triton_python_dir, "triton", "backends", backend_name
        )
        package_data = [
            f"{os.path.relpath(p, backend_path)}/*"
            for p, _, _, in os.walk(backend_path)
        ]

        language_package_data = []
        if language_dir is not None:
            language_package_data = [
                f"{os.path.relpath(p, language_dir)}/*"
                for p, _, _, in os.walk(language_dir)
            ]

        return Backend(
            name=backend_name,
            package_data=package_data,
            language_package_data=language_package_data,
            src_dir=backend_src_dir,
            backend_dir=backend_path,
            language_dir=language_dir,
            install_dir=install_dir,
            is_external=is_external,
        )

    # Copy all in-tree backends under triton/third_party.
    @staticmethod
    def copy(active):
        return [BackendInstaller.prepare(backend) for backend in active]

    # Copy all external plugins provided by the `TRITON_PLUGIN_DIRS` env var.
    # TRITON_PLUGIN_DIRS is a semicolon-separated list of paths to the plugins.
    # Expect to find the name of the backend under dir/backend/name.conf
    @staticmethod
    def copy_externals():
        backend_dirs = os.getenv("TRITON_PLUGIN_DIRS")
        if backend_dirs is None:
            return []
        backend_dirs = backend_dirs.strip().split(";")
        backend_names = [
            Path(os.path.join(dir, "backend", "name.conf")).read_text().strip()
            for dir in backend_dirs
        ]
        return [
            BackendInstaller.prepare(
                backend_name, backend_src_dir=backend_src_dir, is_external=True
            )
            for backend_name, backend_src_dir in zip(backend_names, backend_dirs)
        ]


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_build_type():
    if check_env_flag("DEBUG"):
        return "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        return "RelWithDebInfo"
    elif check_env_flag("TRITON_REL_BUILD_WITH_ASSERTS"):
        return "TritonRelBuildWithAsserts"
    elif check_env_flag("TRITON_BUILD_WITH_O1"):
        return "TritonBuildWithO1"
    else:
        return "Release"


def get_env_with_keys(key: list):
    for k in key:
        if k in os.environ:
            return os.environ[k]
    return ""


def is_offline_build() -> bool:
    """
    Downstream projects and distributions which bootstrap their own dependencies from scratch
    and run builds in offline sandboxes
    may set `TRITON_OFFLINE_BUILD` in the build environment to prevent any attempts at downloading
    pinned dependencies from the internet or at using dependencies vendored in-tree.

    Dependencies must be defined using respective search paths (cf. `syspath_var_name` in `Package`).
    Missing dependencies lead to an early abortion.
    Dependencies' compatibility is not verified.

    Note that this flag isn't tested by the CI and does not provide any guarantees.
    """
    return check_env_flag("TRITON_OFFLINE_BUILD", "")


# --- third party packages -----


class Package(NamedTuple):
    package: str
    name: str
    url: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str


# json
def get_json_package_info():
    url = "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip"
    return Package("json", "", url, "JSON_INCLUDE_DIR", "", "JSON_SYSPATH")


# llvm
def get_llvm_package_info():
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[
            platform.machine()
        ]
    except KeyError:
        arch = platform.machine()
    if system == "Darwin":
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == "arm64":
            system_suffix = "ubuntu-arm64"
        elif arch == "x64":
            vglibc = tuple(map(int, platform.libc_ver()[1].split(".")))
            vglibc = vglibc[0] * 100 + vglibc[1]
            if vglibc > 228:
                # Ubuntu 24 LTS (v2.39)
                # Ubuntu 22 LTS (v2.35)
                # Ubuntu 20 LTS (v2.31)
                system_suffix = "ubuntu-x64"
            elif vglibc > 217:
                # Manylinux_2.28 (v2.28)
                # AlmaLinux 8 (v2.28)
                system_suffix = "almalinux-x64"
            else:
                # Manylinux_2014 (v2.17)
                # CentOS 7 (v2.17)
                system_suffix = "centos-x64"
        else:
            print(
                f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
            )
            return Package(
                "llvm",
                "LLVM-C.lib",
                "",
                "LLVM_INCLUDE_DIRS",
                "LLVM_LIBRARY_DIR",
                "LLVM_SYSPATH",
            )
    else:
        print(
            f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
        )
        return Package(
            "llvm",
            "LLVM-C.lib",
            "",
            "LLVM_INCLUDE_DIRS",
            "LLVM_LIBRARY_DIR",
            "LLVM_SYSPATH",
        )
    # use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    # release_suffix = "assert" if use_assert_enabled_llvm else "release"
    llvm_hash_path = os.path.join(get_triton_root_dir(), "cmake", "llvm-hash.txt")
    with open(llvm_hash_path, "r") as llvm_hash_file:
        rev = llvm_hash_file.read(8)
    name = f"llvm-{rev}-{system_suffix}"
    url = f"https://oaitriton.blob.core.windows.net/public/llvm-builds/{name}.tar.gz"
    return Package(
        "llvm", name, url, "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH"
    )


def open_url(url):
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
    )
    headers = {
        "User-Agent": user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forever.
    return urllib.request.urlopen(request, timeout=300)


# ---- package data ---


def get_triton_cache_path():
    user_home = os.getenv("TRITON_HOME")
    if not user_home:
        user_home = (
            os.getenv("HOME")
            or os.getenv("USERPROFILE")
            or os.getenv("HOMEPATH")
            or None
        )
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


def get_thirdparty_packages(packages: list):
    triton_cache_path = get_triton_cache_path()
    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(triton_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.name)
        if os.environ.get(p.syspath_var_name):
            package_dir = os.environ[p.syspath_var_name]
        version_file_path = os.path.join(package_dir, "version.txt")

        input_defined = p.syspath_var_name in os.environ
        input_exists = os.path.exists(version_file_path)
        input_compatible = input_exists and Path(version_file_path).read_text() == p.url

        if is_offline_build() and not input_defined:
            raise RuntimeError(
                f"Requested an offline build but {p.syspath_var_name} is not set"
            )
        if not is_offline_build() and not input_defined and not input_compatible:
            with contextlib.suppress(Exception):
                shutil.rmtree(package_root_dir)
            os.makedirs(package_root_dir, exist_ok=True)
            print(f"downloading and extracting {p.url} ...")
            with open_url(p.url) as response:
                if p.url.endswith(".zip"):
                    file_bytes = BytesIO(response.read())
                    with zipfile.ZipFile(file_bytes, "r") as file:
                        file.extractall(path=package_root_dir)
                else:
                    with tarfile.open(fileobj=response, mode="r|*") as file:
                        file.extractall(path=package_root_dir)
            # write version url to package_dir
            with open(os.path.join(package_dir, "version.txt"), "w") as f:
                f.write(p.url)
        if p.include_flag:
            thirdparty_cmake_args.append(f"-D{p.include_flag}={package_dir}/include")
        if p.lib_flag:
            thirdparty_cmake_args.append(f"-D{p.lib_flag}={package_dir}/lib")
    return thirdparty_cmake_args

# ---- cmake extension ----


def get_triton_root_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "triton"))


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_triton_root_dir()) / "python" / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


class CMakeClean(clean):

    def initialize_options(self):
        clean.initialize_options(self)
        self.build_temp = get_cmake_dir()


class CMakeBuildPy(build_py):

    def run(self) -> None:
        self.run_command("build_ext")
        return super().run()


class CMakeExtension(Extension):

    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        # self.path = os.path.join(triton_python_dir, path)
        self.path = path

class CMakeBuild(build_ext):

    user_options = build_ext.user_options + [
        ("base-dir=", None, "base directory of Triton")
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.triton_root_dir = get_triton_root_dir()

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        match = re.search(
            r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode()
        )
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def get_pybind11_cmake_args(self):
        pybind11_sys_path = get_env_with_keys(["PYBIND11_SYSPATH"])
        if pybind11_sys_path:
            pybind11_include_dir = os.path.join(pybind11_sys_path, "include")
        else:
            pybind11_include_dir = pybind11.get_include()
        return [f"-DPYBIND11_INCLUDE_DIR={pybind11_include_dir}"]

    def build_extension(self, ext):
        lit_dir = shutil.which("lit")
        ninja_dir = shutil.which("ninja")
        # lit is used by the test suite
        thirdparty_cmake_args = get_thirdparty_packages([get_llvm_package_info()])
        thirdparty_cmake_args += self.get_pybind11_cmake_args()
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # python directories
        python_include_dir = sysconfig.get_path("platinclude")
        cmake_args = [
            "-G",
            "Ninja",  # Ninja is much faster than make
            "-DCMAKE_MAKE_PROGRAM="
            + ninja_dir,  # Pass explicit path to ninja otherwise cmake may cache a temporary path
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DLLVM_ENABLE_WERROR=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DTRITON_BUILD_TUTORIALS=OFF",
            "-DTRITON_BUILD_PYTHON_MODULE=ON",
            "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
            "-DPYTHON_INCLUDE_DIRS=" + python_include_dir,
            "-DTRITON_CODEGEN_BACKENDS="
            + ";".join([b.name for b in _backends if not b.is_external]),
            "-DTRITON_PLUGIN_DIRS="
            + ";".join([b.src_dir for b in _backends if b.is_external]),
        ]
        if lit_dir is not None:
            cmake_args.append("-DLLVM_EXTERNAL_LIT=" + lit_dir)
        cmake_args.extend(thirdparty_cmake_args)

        # configuration
        cfg = get_build_type()
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
        build_args += ["-j" + max_jobs]

        if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD"):
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld",
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]

        if check_env_flag("TRITON_BUILD_WITH_CCACHE"):
            cmake_args += [
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]
        cmake_args += ["-DTRITON_BUILD_PROTON=OFF"]
        cmake_args_append = os.getenv("TRITON_APPEND_CMAKE_ARGS")
        if cmake_args_append is not None:
            cmake_args += shlex.split(cmake_args_append)

        env = os.environ.copy()
        cmake_dir = get_cmake_dir()
        subprocess.check_call(
            ["cmake", script_dir] + cmake_args, cwd=cmake_dir, env=env
        )
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)


def get_platform_dependent_src_path(subdir):
    return lambda platform, version: (
        (
            lambda version_major, version_minor1, version_minor2,: (
                f"targets/{platform}/{subdir}"
                if int(version_major) >= 12 and int(version_minor1) >= 5
                else subdir
            )
        )(*version.split("."))
    )


_backends = [*BackendInstaller.copy([]), *BackendInstaller.copy_externals()]


def add_link_to_backends(backends):
    for backend in backends:
        if os.path.islink(backend.install_dir):
            os.unlink(backend.install_dir)
        if os.path.exists(backend.install_dir):
            shutil.rmtree(backend.install_dir)
        os.symlink(backend.backend_dir, backend.install_dir)

        if backend.language_dir:
            # Link the contents of each backend's `language` directory into
            # `triton.language.extra`.
            extra_dir = os.path.abspath(
                os.path.join(triton_python_dir, "triton", "language", "extra")
            )
            for x in os.listdir(backend.language_dir):
                src_dir = os.path.join(backend.language_dir, x)
                install_dir = os.path.join(extra_dir, x)
                if os.path.islink(install_dir):
                    os.unlink(install_dir)
                if os.path.exists(install_dir):
                    shutil.rmtree(install_dir)
                os.symlink(src_dir, install_dir)


def add_links():
    add_link_to_backends(_backends)


def insert_at_file_start(filepath, import_lines):
    import tempfile
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if import_lines in content:
            return False
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write(import_lines + '\n\n')
            with open(filepath, 'r') as original_file:
                tmp_file.write(original_file.read())
        backup_path = filepath + '.bak'
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.move(filepath, backup_path)
        shutil.move(tmp_file.name, filepath)
        print(f"[INFO]: {filepath} is patched")
        return True
    except PermissionError:
        print(f"[ERROR]: No permission to write to {filepath}!")
    except FileNotFoundError:
        print(f"[ERROR]: {filepath} does not exist!")
    except Exception as e:
        print(f"[ERROR]: Unknown error: {str(e)}")
    return False

def append_at_file_end(filepath, import_lines):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if import_lines in content:
            return False
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write('\n' + import_lines)
        return True
    except PermissionError:
        print(f"[ERROR]: No permission to write to {filepath}!")
    except FileNotFoundError:
        print(f"[ERROR]: {filepath} does not exist!")
    except Exception as e:
        print(f"[ERROR]: Unknown error: {str(e)}")
    return False

def post_install(self):
    install_dir = os.path.join(self.install_lib, "triton")
    init_path = os.path.join(install_dir, "__init__.py")
    patched_content = f"""
import sys
from .triton_patch.language import _utils as ascend_utils
sys.modules['triton.language._utils'] = ascend_utils
from .triton_patch.compiler import compiler as ascend_compiler
sys.modules['triton.compiler.compiler'] = ascend_compiler
from .triton_patch.compiler import code_generator as ascend_code_generator
sys.modules['triton.compiler.code_generator'] = ascend_code_generator
from .triton_patch.compiler import errors as ascend_errors
sys.modules['triton.compiler.errors'] = ascend_errors
from .triton_patch.runtime import autotuner as ascend_autotuner
sys.modules['triton.runtime.autotuner'] = ascend_autotuner
from .triton_patch import testing as ascend_testing
sys.modules['triton.testing'] = ascend_testing
"""
    insert_at_file_start(init_path, patched_content)

    content_to_append = f"""
from .triton_patch.language.core import dot, gather, insert, subview
from .triton_patch.language.standard import flip, sigmoid, softmax
from .triton_patch.language.math import umulhi, exp, exp2, log, log2, cos, sin, sqrt, sqrt_rn, rsqrt, div_rn, erf, tanh, floor, ceil
from . import language

language.dot = dot
language.flip = flip
language.sigmoid = sigmoid
language.softmax = softmax
language.gather = gather
language.insert = insert
language.subview = subview

# from .triton_patch.language.core import dtype, pointer_type, block_type, function_type
# language.core.dtype = dtype
# language.core.pointer_type = pointer_type
# language.core.block_type = block_type
# language.core.function_type = function_type

from .triton_patch.language.semantic import arange, floordiv, atom_red_typechecking_impl, \
        atomic_max, atomic_min, maximum, minimum
language.semantic.arange = arange
language.semantic.floordiv = floordiv
language.semantic.atom_red_typechecking_impl = atom_red_typechecking_impl
language.semantic.atomic_max = atomic_max
language.semantic.atomic_min = atomic_min
language.semantic.maximum = maximum
language.semantic.minimum = minimum

language.umulhi = umulhi
language.exp = exp
language.exp2 = exp2
language.log = log
language.log2 = log2
language.cos = cos
language.sin = sin
language.sqrt = sqrt
language.sqrt_rn = sqrt_rn
language.rsqrt = rsqrt
language.div_rn = div_rn
language.erf = erf
language.tanh = tanh
language.floor = floor
language.ceil = ceil
language.math.umulhi = umulhi
language.math.exp = exp
language.math.exp2 = exp2
language.math.log = log
language.math.log2 = log2
language.math.cos = cos
language.math.sin = sin
language.math.sqrt = sqrt
language.math.sqrt_rn = sqrt_rn
language.math.rsqrt = rsqrt
language.math.div_rn = div_rn
language.math.erf = erf
language.math.tanh = tanh
language.math.floor = floor
language.math.ceil = ceil
language.math.isnan = language.extra.ascend.libdevice.isnan
language.math.isinf = language.extra.ascend.libdevice.isinf
"""
    append_at_file_end(init_path, content_to_append)

class plugin_install(install):

    def run(self):
        add_links()
        install.run(self)
        post_install(self)

class plugin_develop(develop):

    def run(self):
        assert False, "[ERROR] develop mode is unsupported for now"
        add_links()
        develop.run(self)
        post_install(self)

class plugin_bdist_wheel(bdist_wheel):

    def run(self):
        add_links()
        bdist_wheel.run(self)

class plugin_egginfo(egg_info):

    def run(self):
        add_links()
        egg_info.run(self)


_package_data = {
    "triton/tools": ["compile.h", "compile.c"],
    **{f"triton/backends/{b.name}": b.package_data for b in _backends},
    "triton/language/extra": sum((b.language_package_data for b in _backends), []),
}

def get_language_extra_packages():
    packages = []
    for backend in _backends:
        if backend.language_dir is None:
            continue

        # Walk the `language` directory of each backend to enumerate
        # any subpackages, which will be added to `triton.language.extra`.
        for dir, dirs, files in os.walk(backend.language_dir, followlinks=True):
            if (
                not any(f for f in files if f.endswith(".py"))
                or dir == backend.language_dir
            ):
                # Ignore directories with no python files.
                # Also ignore the root directory which corresponds to
                # "triton/language/extra".
                continue
            subpackage = os.path.relpath(dir, backend.language_dir)
            package = os.path.join("triton/language/extra", subpackage)
            packages.append(package)

    return list(packages)


def get_packages(backends):
    packages = [
        "triton",
        "triton/_C",
        "triton/compiler",
        "triton/language",
        "triton/language/extra",
        "triton/runtime",
        "triton/backends",
        "triton/tools",
    ]
    packages += [f"triton/backends/{backend.name}" for backend in backends]
    packages += get_language_extra_packages()
    packages += [
        "triton/triton_patch",
        "triton/triton_patch/language",
        "triton/triton_patch/compiler",
        "triton/triton_patch/runtime",
    ]
    return packages

def get_package_dir(backends):
    triton_root_rel_dir = "triton/python/triton"
    package_dir = {
        "triton": f"{triton_root_rel_dir}",
        "triton/_C": f"{triton_root_rel_dir}/_C",
        "triton/backends": f"{triton_root_rel_dir}/backends",
        "triton/compiler": f"{triton_root_rel_dir}/compiler",
        "triton/language": f"{triton_root_rel_dir}/language",
        "triton/language/extra": f"{triton_root_rel_dir}/language/extra",
        "triton/runtime": f"{triton_root_rel_dir}/runtime",
        "triton/tools": f"{triton_root_rel_dir}/tools",
    }
    for backend in backends:
        package_dir[f"triton/backends/{backend.name}"] = f"{triton_root_rel_dir}/backends/{backend.name}"
    language_extra_list = get_language_extra_packages()
    for extra_full in language_extra_list:
        extra_name = extra_full.replace("triton/language/extra/", "")
        package_dir[extra_full] = f"{triton_root_rel_dir}/language/extra/{extra_name}"
    #
    triton_patch_root_rel_dir = "triton_patch/python/triton_patch"
    package_dir["triton/triton_patch"] = f"{triton_patch_root_rel_dir}"
    package_dir["triton/triton_patch/language"] = f"{triton_patch_root_rel_dir}/language"
    package_dir["triton/triton_patch/compiler"] = f"{triton_patch_root_rel_dir}/compiler"
    package_dir["triton/triton_patch/runtime"] = f"{triton_patch_root_rel_dir}/runtime"
    return package_dir

def get_entry_points():
    entry_points = {}
    return entry_points


def get_git_commit_hash(length=8):
    try:
        triton_root = os.getcwd()
        triton_ascend_root = os.environ.get("TRITON_PLUGIN_DIRS", triton_root)
        os.chdir(triton_ascend_root)
        cmd = ["git", "rev-parse", f"--short={length}", "HEAD"]
        git_commit_hash = subprocess.check_output(cmd).strip().decode("utf-8")
        os.chdir(triton_root)
        return "+git{}".format(git_commit_hash)
    except Exception:
        return ""


setup(
    name=os.environ.get("TRITON_WHEEL_NAME", "triton"),
    version=os.environ.get("TRITON_VERSION", "3.2.0")
    + get_git_commit_hash()
    + os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", ""),
    author="",
    author_email="",
    description="A language and compiler for custom Deep Learning operations on Huawei hardwares",
    long_description="",
    package_dir=get_package_dir(_backends),
    packages=get_packages(_backends),
    entry_points=get_entry_points(),
    package_data=_package_data,
    include_package_data=True,
    ext_modules=[CMakeExtension("triton", "triton/_C/")],
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": CMakeBuildPy,
        "clean": CMakeClean,
        "install": plugin_install,
        "develop": plugin_develop,
        "bdist_wheel": plugin_bdist_wheel,
        "egg_info": plugin_egginfo,
    },
    zip_safe=False,
    # for PyPI
    keywords=["Compiler", "Deep Learning"],
    url="https://gitee.com/ascend/triton-ascend/",
    extras_require={
        "build": [
            "cmake>=3.20",
            "lit",
        ]
    },
)
