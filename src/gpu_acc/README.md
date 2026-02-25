# GPU 加速模块 (gpu_acc)

## 目录结构（重组后）

```
gpu_acc/
├── README.md                 # 本说明
├── ob_gpu_config.h/cpp       # 全局配置常量（GPU 阈值等）
├── CMakeLists.txt
│
├── metal/                    # Metal 运行时与各功能实现
│   ├── ob_metal_context.h/mm # 共享：Device、CommandQueue、init、错误信息
│   ├── vector/
│   │   ├── ob_metal_l2.h/mm      # L2 距离
│   │   └── ob_metal_cosine.h/mm  # 余弦距离
│   └── count_distinct/
│       └── ob_gpu_count_distinct.h/mm  # Count Distinct
│
├── shader/                   # Metal 着色器源码（.metal）
│   ├── vector_l2.metal
│   ├── vector_cosine.metal
│   └── count_distinct_kernel.metal
│
└── encoder/                  # 编码器封装（Command Encoder 调度）
    └── ob_metal_encoder.h/mm  # 通用：dispatch 一维 compute kernel、同步
```

## 可抽象与复用点

### 1. Metal 上下文 (metal/ob_metal_context)

- **现状**：`ob_metal_l2.mm`、`ob_metal_cosine.mm`、`ob_gpu_count_distinct.mm` 各自维护一套：
  - `id<MTLDevice>`, `id<MTLCommandQueue>`
  - `init_metal()`：创建设备、队列、编译 library
  - `g_metal_init_error[]`、`is_metal_ready()` 等
- **抽象**：统一使用 `ObMetalContext`（或单例）提供：
  - `bool init()`
  - `id<MTLDevice> device()`, `id<MTLCommandQueue> queue()`
  - `const char* last_error()`
- **继承/复用**：L2、Cosine、CountDistinct 只负责创建自己的 PipelineState，Device/Queue 从 Context 获取。

### 2. Buffer 创建与页对齐

- **现状**：三处都有 `is_page_aligned()` 和“指针页对齐则 NoCopy，否则 WithBytes”的逻辑。
- **抽象**：在 `ob_metal_context` 或 `encoder` 中提供：
  - `id<MTLBuffer> buffer_from_ptr(id<MTLDevice> device, const void* ptr, size_t len, bool read_only)`
  - 内部封装页对齐判断与 `newBufferWithBytesNoCopy` / `newBufferWithBytes`。

### 3. Encoder 调度 (encoder/ob_metal_encoder)

- **现状**：各 .mm 中重复：
  - `[queue commandBuffer]` → `[cmd computeCommandEncoder]` → `setComputePipelineState` → `setBuffer/setBytes` → `dispatchThreads` → `endEncoding` → `commit` → `waitUntilCompleted`
- **抽象**：提供辅助函数，例如：
  - `run_compute_kernel(queue, pipeline, buffers_and_offsets[], bytes[], grid_size)`  
  或 C 风格：传入 (queue, pipeline, buffer_count, buffer_ptrs, byte_count, byte_ptrs, grid_len)，内部完成 encode + dispatch + commit + wait。

### 4. 向量距离接口（可选）

- **现状**：L2 与 Cosine 接口类似（query, vectors, count, dim, distances_out），另有 strided、batch、f64 等变体。
- **抽象**：可定义公共接口，例如：
  - `IVectorDistanceKernel`：`compute_contiguous()`, `compute_strided()` 等；
  - L2 / Cosine 各实现一份，便于后续扩展 IP、其他距离。
- **当前**：先做目录与 Context/Encoder 抽象，接口层可后续再统一。

### 5. Shader 源码归属

- **现状**：L2、Cosine、CountDistinct 的 kernel 源码均内嵌在 .mm 的字符串中；`metal/ob_count_distinct_kernel.metal` 存在但未被使用。
- **重组**：将全部 .metal 源码放入 `shader/` 目录，作为唯一来源；.mm 在运行时通过 `[device newLibraryWithSource:...]` 或编译期包含字符串读取。当前保留 .mm 内嵌实现以保证构建不变，shader/ 作为规范来源便于后续改为从文件加载或编译成 .metallib。

## 依赖与编译

- 仅 Apple 平台启用 Metal（`if(APPLE)`）；其他平台不编译 metal/、encoder/ 及 shader。
- 对外头文件保持原有路径或通过 `gpu_acc/ob_gpu_config.h`、`gpu_acc/metal/vector/ob_metal_l2.h` 等引用，保证 `ob_expr_vector.cpp`、`ob_vector_kmeans_ctx.cpp`、aggregate 等调用方无需改动路径（或仅做最小 include 调整）。
