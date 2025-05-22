#ifndef TRITON_DEVICE_PRINT_H
#define TRITON_DEVICE_PRINT_H

#include "experiment/runtime/runtime/rt.h"
#include "stdio.h"
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <type_traits>

#define LogBufferPaddingBytes 64
#define BlockMaxSize 16 * 1024
#define VerifyBorder(nextField, maxBuf)                                        \
  if (nextField > maxBuf) {                                                    \
    std::cout << ("\nWARNING: out of bound! try best to print\n");             \
    return;                                                                    \
  }
#define __gm__

namespace TTAscDebug {

enum NodeTy { END, NORMAL, FLOAT, INT, CHAR, STRING, POINTER };

struct PrintPayloadData {
  __gm__ char *LogWholeRegion;
  unsigned BlockNum;
  size_t LogBufferSize;
  PrintPayloadData()
      : LogWholeRegion((__gm__ char *)nullptr), LogBufferSize(0), BlockNum(0) {}
};

struct DebugTunnelData {
  PrintPayloadData PrintData;
  DebugTunnelData() {}
};

void PrintFormatString(int8_t *&buf, int8_t *maxbuf) {
  // 读取长度
  short len = *reinterpret_cast<short *>(buf);
  buf += sizeof(len);

  // 检查缓冲区边界
  if (buf + len > maxbuf) {
    throw std::runtime_error("Buffer overflow");
  }

  // 获取格式字符串并输出
  const char *str = reinterpret_cast<const char *>(buf);
  std::cout << str; // 直接使用 std::cout 输出字符串

  // 移动缓冲区指针
  buf += len;
}

template <typename T>
void PrintFormatString(int8_t *&buf, int8_t *maxbuf, T param) {
  // 读取长度
  short len = *reinterpret_cast<short *>(buf);
  buf += sizeof(len);

  // 确保不会越界
  if (buf + len > maxbuf) {
    throw std::runtime_error("Buffer overflow");
  }

  // 获取格式字符串
  const char *fmt = reinterpret_cast<const char *>(buf);
  buf += len;

  // 处理格式字符串并输出
  bool in_format = false;
  for (int i = 0; i < len; ++i) {
    if (fmt[i] == '%') {
      if (in_format) {
        std::cout << '%'; // 遇到 %%
        in_format = false;
      } else {
        in_format = true;
      }
    } else {
      if (in_format) {
        // 处理格式说明符
        switch (fmt[i]) {
        case 'd':
        case 'i':
          if constexpr (std::is_convertible_v<T, int>) {
            std::cout << static_cast<int>(param);
          } else {
            std::cerr << "Error: %d|i is invalid for typename\n";
          }
          break;
        case 'u':
          if constexpr (std::is_convertible_v<T, unsigned int>) {
            std::cout << static_cast<unsigned int>(param);
          } else {
            std::cerr << "Error: %u is invalid for typename\n";
          }
          break;
        case 'f':
        case 'F':
          if constexpr (std::is_convertible_v<T, double>) {
            std::cout << static_cast<double>(param);
          } else {
            std::cerr << "Error: %f|F is invalid for typename\n";
          }
          break;
        case 'e':
        case 'E':
          if constexpr (std::is_convertible_v<T, double>) {
            std::cout << std::scientific << static_cast<double>(param);
          } else {
            std::cerr << "Error: %e|E is invalid for typename\n";
          }
          break;
        case 'g':
        case 'G':
          if constexpr (std::is_convertible_v<T, double>) {
            std::cout << std::defaultfloat << static_cast<double>(param);
          } else {
            std::cerr << "Error: %g|G is invalid for typename\n";
          }
          break;
        case 'x':
        case 'X':
          if constexpr (std::is_convertible_v<T, int>) {
            std::cout << std::hex << static_cast<int>(param);
          } else {
            std::cerr << "Error: %x|X is invalid for typename\n";
          }
          break;
        case 'o':
          if constexpr (std::is_convertible_v<T, int>) {
            std::cout << std::oct << static_cast<int>(param);
          } else {
            std::cerr << "Error: %o is invalid for typename\n";
          }
          break;
        case 'c':
          if constexpr (std::is_convertible_v<T, char>) {
            std::cout << static_cast<char>(param);
          } else {
            std::cerr << "Error: %c is invalid for typename\n";
          }
          break;
        case 's':
          if constexpr (std::is_convertible_v<T, const char *>) {
            std::cout << static_cast<const char *>(param);
          } else {
            std::cerr << "Error: %s is invalid for typename\n";
          }
          break;
        case 'p':
          if constexpr (std::is_convertible_v<T, void *>) {
            std::cout << reinterpret_cast<void *>(param);
          } else {
            std::cerr << "Error: %p is invalid for typename\n";
          }
          break;
        default:
          std::cout << '%' << fmt[i]; // 无效格式说明符
          break;
        }
        in_format = false;
      } else {
        std::cout << fmt[i];
      }
    }
  }
}

void AnalyzeSerializedData(int8_t *buf, int logSize, int maxSize) {
  int8_t *bufEndAddr = buf + logSize;
  int8_t *maxbuf = buf + maxSize;
  while (buf < bufEndAddr) {
    VerifyBorder((buf + sizeof(int8_t)), maxbuf);
    int8_t type = *(int8_t *)buf;
    while (type != NodeTy::END) {
      buf += sizeof(type);
      switch (type) {
      default:
        break;
      case NodeTy::NORMAL: {
        PrintFormatString(buf, maxbuf);
        break;
      }
      case NodeTy::FLOAT: {
        VerifyBorder((buf + sizeof(float)), maxbuf);
        float param = *(float *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case NodeTy::INT: {
        VerifyBorder((buf + sizeof(long long int)), maxbuf);
        long long int param = *(long long int *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case NodeTy::STRING: {
        VerifyBorder((buf + sizeof(short)), maxbuf);
        short strlen = *(short *)buf;
        buf += sizeof(strlen);
        VerifyBorder((buf + strlen), maxbuf);
        char *param = reinterpret_cast<char *>(buf);
        buf += strlen;
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case NodeTy::CHAR: {
        VerifyBorder((buf + sizeof(char)), maxbuf);
        char param = *(char *)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      case NodeTy::POINTER: {
        VerifyBorder((buf + 8), maxbuf);
        void *param = *(void **)buf;
        buf += sizeof(param);
        PrintFormatString(buf, maxbuf, param);
        break;
      }
      }
      VerifyBorder((buf + sizeof(int8_t)), maxbuf);
      type = *(int8_t *)buf;
    }
    buf += 1;
  }
}

void OnHostInitialize(PrintPayloadData *PrintData, unsigned BlockNum) {
  PrintData->LogBufferSize = BlockMaxSize;
  PrintData->BlockNum = BlockNum;
  int WholeSize =
      (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum;

  void *Hbm_PrintPayloadData_start_addr = NULL;
  // Not sure how to use the module_id param of rtMalloc
  uint16_t ModuleId = 0;
  rtError_t error =
      rtMalloc(reinterpret_cast<void **>(&Hbm_PrintPayloadData_start_addr),
               WholeSize, RT_MEMORY_HBM, ModuleId);
  if (error != RT_ERROR_NONE) {
    std::cout
        << ("ERROR:The memory for the printing function on the device side "
            "fails to be allocated.");
    std::cout << ("As a result, the printing function fails!\n");
    return;
  }
  PrintData->LogWholeRegion = (__gm__ char *)Hbm_PrintPayloadData_start_addr;
}

void OnHostFinish(PrintPayloadData *PrintData, rtStream_t Stream) {
  if (!PrintData->LogWholeRegion) {
    return;
  }
  std::size_t WholeSize =
      (PrintData->LogBufferSize + LogBufferPaddingBytes) * PrintData->BlockNum;
  char *hostMemOut2;
  // Not sure how to use the module_id param of rtMalloc
  uint16_t ModuleId = 0;
  rtError_t error = rtMallocHost(reinterpret_cast<void **>(&hostMemOut2),
                                 WholeSize, ModuleId);
  if (error != RT_ERROR_NONE) {
    std::cout
        << ("ERROR:The memory for the printing function on the device side "
            "fails to be allocated.");
    std::cout << ("As a result, the printing function fails!\n");
    return;
  }
  error = rtMemcpyAsync(hostMemOut2, WholeSize, PrintData->LogWholeRegion,
                        WholeSize, RT_MEMCPY_DEVICE_TO_HOST, Stream);
  if (error != RT_ERROR_NONE) {
    std::cout << ("ERROR: The memory copy of the device print on fails,");
    std::cout << ("and the printing function is invalid!\n");
    return;
  }
  error = rtStreamSynchronize(Stream);
  if (error != RT_ERROR_NONE) {
    std::cout << ("ERROR: Synchronous waiting for the device print failed.\n");
    std::cout << ("The printing function is invalid!\n");
    return;
  }
  char *outRaw2 = static_cast<char *>(hostMemOut2);
  const char *Line = "-------------------------------------------------------";
  // Precheck if any print data is ready
  for (int B = 0; B < PrintData->BlockNum; B++) {
    char *Log =
        (outRaw2 + (PrintData->LogBufferSize + LogBufferPaddingBytes) * B);
    size_t LogSize = *reinterpret_cast<size_t *>(Log);
    if (LogSize > 0 && LogSize <= PrintData->LogBufferSize) {
      std::cout << "LogBufferSize of each core is : "
                << PrintData->LogBufferSize << "Bytes\n";
      std::cout << Line << "\n";
      std::cout << ("----------------------HiIPU "
                    "Print----------------------\n");
      std::cout << Line << "\n";
      break;
    }
  }

  for (int B = 0; B < PrintData->BlockNum; B++) {
    char *Log =
        (outRaw2 + (PrintData->LogBufferSize + LogBufferPaddingBytes) * B);
    size_t LogSize = *reinterpret_cast<size_t *>(Log);
    if (LogSize < 0 || LogSize > PrintData->LogBufferSize) {
      std::cout << (" LOG SIZE ERROR !!! \n");
      std::cout << " log size needed = " << LogSize;
      std::cout << " , buf size = " << PrintData->LogBufferSize << "\n";
      LogSize = PrintData->LogBufferSize;
      continue;
    }
    if (LogSize == 0) {
      continue;
    }
    std::cout << "==> Block " << B << ", LogSize = " << LogSize << " Bytes\n";
    int8_t *Buf = reinterpret_cast<int8_t *>(Log + LogBufferPaddingBytes);
    AnalyzeSerializedData(Buf, LogSize, PrintData->LogBufferSize);
    std::cout << ("\n");
    std::cout << Line << "\n";
  }
  error = rtFree(PrintData->LogWholeRegion);
  if (error != RT_ERROR_NONE) {
    std::cout << ("ERROR: The memory free of the device print fails\n");
    return;
  }
  error = rtFreeHost(hostMemOut2);
  if (error != RT_ERROR_NONE) {
    std::cout << ("ERROR: The memory free of the device print fails\n");
    return;
  }
}

DebugTunnelData *Open(unsigned BlockNum) {
  DebugTunnelData debugTunnelDataForHost;
  OnHostInitialize(&(debugTunnelDataForHost.PrintData), BlockNum);
  void *Hbm_PrintPayloadData_start_addr = NULL;
  // Not sure how to use the module_id param of rtMalloc
  uint16_t ModuleId = 0;
  rtError_t error =
      rtMalloc(reinterpret_cast<void **>(&Hbm_PrintPayloadData_start_addr),
               sizeof(debugTunnelDataForHost), RT_MEMORY_HBM, ModuleId);
  if (error != RT_ERROR_NONE) {
    std::cout
        << ("ERROR: The memory for the printing function on the device side "
            "fails to be allocated.");
    std::cout << ("As a result, the printing function fails!\n");
    return nullptr;
  }
  if (Hbm_PrintPayloadData_start_addr == nullptr) {
    std::cout << ("WARNING: failed to allocate DebugTunnelData memory\n");
    return nullptr;
  }
  error = rtMemcpy(Hbm_PrintPayloadData_start_addr,
                   sizeof(debugTunnelDataForHost), &debugTunnelDataForHost,
                   sizeof(debugTunnelDataForHost), RT_MEMCPY_HOST_TO_DEVICE);
  if (error != RT_ERROR_NONE) {
    std::cout << ("ERROR: The memory copy of the device print on fails, ");
    std::cout << ("and the printing function is invalid!\n");
    return nullptr;
  }
  return reinterpret_cast<DebugTunnelData *>(Hbm_PrintPayloadData_start_addr);
}

void Close(DebugTunnelData *DTData, rtStream_t Stream) {
  if (!DTData) {
    return;
  }
  DebugTunnelData debugTunnelDataForHost;
  rtError_t error = rtStreamSynchronize(Stream);
  if (error != RT_ERROR_NONE) {
    std::cout << ("ERROR: Synchronous waiting for the device print failed.\n");
    std::cout << ("The printing function is invalid!\n");
  }
  error =
      rtMemcpy(&debugTunnelDataForHost, sizeof(debugTunnelDataForHost), DTData,
               sizeof(debugTunnelDataForHost), RT_MEMCPY_DEVICE_TO_HOST);
  if (error != RT_ERROR_NONE) {
    std::cout << ("ERROR: The memory copy of the device print on fails, ");
    std::cout << ("and the printing function is invalid!\n");
    return;
  }
  OnHostFinish(&(debugTunnelDataForHost.PrintData), Stream);

  error = rtFree(DTData);
  if (error != RT_ERROR_NONE) {
    std::cout << ("ERROR: The memory free of the device print fails, ");
    std::cout << ("and the device print is invalid!\n");
    return;
  }
  fflush(stdout);
}

} // namespace TTAscDebug

#endif
