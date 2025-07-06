# Triton Ascend

![Triton Ascend](https://img.shields.io/badge/Triton%20Ascend-v1.0-blue)

Triton is a programming language and compiler designed for efficiently writing custom deep learning primitives. Its goal is to create an open-source environment that allows developers to write code efficiently while offering more flexibility than existing domain-specific languages (DSLs).

Triton-Ascend focuses on the Ascend platform, enabling Triton code to run efficiently on Ascend hardware. 

This document provides two installation methods to meet different user needs. You can choose the most suitable method based on your specific requirements.

## Table of Contents

- [Installation Methods](#installation-methods)
  - [Python Wheel Installation](#python-wheel-installation)
  - [Source Code Compilation Installation](#source-code-compilation-installation)
- [Environment Preparation](#environment-preparation)
  - [Python Version Requirements](#python-version-requirements)
  - [Installing Ascend CANN](#installing-ascend-cann)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)
- [Releases](#releases)

## Installation Methods

### Python Wheel Installation

The fastest and simplest way to install Triton-Ascend is through the Python Wheel package. This method is ideal for users who want to quickly deploy Triton-Ascend without any hassle.

To install via Python Wheel, follow these steps:

1. Ensure you have Python 3.9 to 3.11 installed on your system.
2. Open your terminal or command prompt.
3. Run the following command:

   ```bash
   pip install triton-ascend
   ```

### Source Code Compilation Installation

If you need to develop or customize Triton-Ascend, you should opt for the source code compilation method. This approach allows you to modify the source code as per your project requirements and compile a customized version of Triton-Ascend.

To install via source code, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/tchu10/triton-ascend.git
   ```

2. Navigate to the cloned directory:

   ```bash
   cd triton-ascend
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Compile the source code:

   ```bash
   python setup.py install
   ```

5. Verify the installation by running:

   ```bash
   python -c "import triton; print(triton.__version__)"
   ```

## Environment Preparation

### Python Version Requirements

Triton-Ascend requires Python version **3.9 to 3.11**. Ensure you have a compatible version installed before proceeding with the installation.

### Installing Ascend CANN

The Compute Architecture for Neural Networks (CANN) is a heterogeneous computing architecture launched by Ascend for AI scenarios. It supports various AI frameworks, including MindSpore, PyTorch, and TensorFlow, while serving AI processors and programming. CANN plays a crucial role in enhancing the computational efficiency of Ascend AI processors.

To install CANN, visit the Ascend community website and follow their software installation guidelines.

During the installation process, select CANN version **8.2.RC1.alpha002** and specify the CPU architecture (AArch64/X86_64) and NPU hardware model (910b) based on your actual environment.

- Community download link: [Download CANN](https://www.hiascend.com/developer/download/community/result?module=cann)
- Community installation guide link: [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=)

## Usage

Once you have installed Triton-Ascend, you can start using it to develop your deep learning applications. Here are some basic usage examples:

### Example 1: Basic Triton Kernel

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, N):
    pid = tl.program_id(0)
    offsets = pid * tl.numel(tl.arange(N))
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(z_ptr + offsets, x + y)

def add(x, y):
    N = x.size
    z = torch.empty_like(x)
    add_kernel[(N + 255) // 256](x.data_ptr(), y.data_ptr(), z.data_ptr(), N)
    return z
```

### Example 2: Running a Simple Model

```python
import torch
import triton

# Define your model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# Instantiate and use the model
model = SimpleModel()
input_data = torch.randn(1, 10)
output_data = model(input_data)
print(output_data)
```

## Contributing

We welcome contributions to Triton-Ascend. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Create a pull request detailing your changes.

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

Triton-Ascend is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Support

If you encounter any issues or have questions, please open an issue in the GitHub repository. We will do our best to assist you.

## Releases

For the latest releases, visit the [Releases](https://github.com/tchu10/triton-ascend/releases) section of the repository. Here, you can find downloadable files and release notes for each version. 

This link provides access to the latest updates and features. Be sure to check it regularly for new releases.