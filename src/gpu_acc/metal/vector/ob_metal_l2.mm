/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gpu_acc/metal/vector/ob_metal_l2.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <sys/param.h>
#define METAL_PAGE_SIZE (static_cast<size_t>(PAGE_SIZE))
#else
#define METAL_PAGE_SIZE 4096u
#endif

namespace oceanbase
{
namespace gpu_acc
{
namespace vector_metal
{

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static id<MTLComputePipelineState> g_pipeline_l2 = nil;
static id<MTLComputePipelineState> g_pipeline_l2_strided = nil;
static id<MTLComputePipelineState> g_pipeline_l2_strided_i32 = nil;
static id<MTLComputePipelineState> g_pipeline_l2_batch = nil;
static bool g_metal_initialized = false;
static bool g_metal_init_failed = false;
static char g_metal_init_error[256] = {0};

static bool is_page_aligned(const void *ptr)
{
  return (reinterpret_cast<uintptr_t>(ptr) & (METAL_PAGE_SIZE - 1)) == 0;
}

static bool init_metal()
{
  if (g_metal_initialized) return true;
  if (g_metal_init_failed) return false;

  @autoreleasepool {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
      NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
      if (devices.count > 0) {
        g_device = devices[0];
      }
    }
    if (!g_device) {
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "no Metal device");
      g_metal_init_failed = true;
      return false;
    }

    NSError *error = nil;
    NSString *source = @R"metal(
#include <metal_stdlib>
using namespace metal;

// L2 squared: contiguous layout vectors[i*dim+j]
kernel void metal_l2_sqr(
    device const float *query   [[buffer(0)]],
    device const float *vectors [[buffer(1)]],
    device float *distances     [[buffer(2)]],
    constant uint &dim          [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint d = dim;
  device const float *v = vectors + gid * d;
  float sum = 0.0f;
  for (uint j = 0; j < d; j++) {
    float diff = query[j] - v[j];
    sum += diff * diff;
  }
  distances[gid] = sum;
}

// Batch L2 squared: M queries, N vectors -> M*N distances. gid in [0, M*N), row = gid/N, col = gid%N.
kernel void metal_l2_sqr_batch(
    device const float *queries  [[buffer(0)]],
    device const float *vectors  [[buffer(1)]],
    device float *distances      [[buffer(2)]],
    constant uint &dim          [[buffer(3)]],
    constant uint &num_vectors  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint d = dim;
  uint nv = num_vectors;
  uint row = gid / nv;
  uint col = gid % nv;
  device const float *q = queries + row * d;
  device const float *v = vectors + col * d;
  float sum = 0.0f;
  for (uint j = 0; j < d; j++) {
    float diff = q[j] - v[j];
    sum += diff * diff;
  }
  distances[gid] = sum;
}

// L2 squared: offset-based layout, row i at vectors_data + byte_offsets[i] (no CPU copy), float elements
kernel void metal_l2_sqr_strided(
    device const float *query       [[buffer(0)]],
    device const char *vectors_data  [[buffer(1)]],
    device const uint *byte_offsets  [[buffer(2)]],
    device float *distances          [[buffer(3)]],
    constant uint &dim               [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint d = dim;
  device const float *v = (device const float *)(vectors_data + byte_offsets[gid]);
  float sum = 0.0f;
  for (uint j = 0; j < d; j++) {
    float diff = query[j] - v[j];
    sum += diff * diff;
  }
  distances[gid] = sum;
}

// L2 squared: int32 elements, convert to float in kernel; query and vectors_data as int32_t*
kernel void metal_l2_sqr_strided_i32(
    device const int32_t *query     [[buffer(0)]],
    device const char *vectors_data [[buffer(1)]],
    device const uint *byte_offsets [[buffer(2)]],
    device float *distances         [[buffer(3)]],
    constant uint &dim              [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint d = dim;
  device const int32_t *v = (device const int32_t *)(vectors_data + byte_offsets[gid]);
  float sum = 0.0f;
  for (uint j = 0; j < d; j++) {
    float q = static_cast<float>(query[j]);
    float vf = static_cast<float>(v[j]);
    float diff = q - vf;
    sum += diff * diff;
  }
  distances[gid] = sum;
}
)metal";

    g_library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!g_library) {
      const char *msg = error ? [[error localizedDescription] UTF8String] : "library compile failed";
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "library: %.200s", msg ? msg : "(nil)");
      g_metal_init_failed = true;
      return false;
    }

    id<MTLFunction> func = [g_library newFunctionWithName:@"metal_l2_sqr"];
    if (!func) {
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "function metal_l2_sqr not found");
      g_metal_init_failed = true;
      return false;
    }
    g_pipeline_l2 = [g_device newComputePipelineStateWithFunction:func error:&error];
    if (!g_pipeline_l2) {
      const char *msg = error ? [[error localizedDescription] UTF8String] : "pipeline failed";
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "pipeline: %.200s", msg ? msg : "(nil)");
      g_metal_init_failed = true;
      return false;
    }

    id<MTLFunction> func_strided = [g_library newFunctionWithName:@"metal_l2_sqr_strided"];
    if (!func_strided) {
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "function metal_l2_sqr_strided not found");
      g_metal_init_failed = true;
      return false;
    }
    g_pipeline_l2_strided = [g_device newComputePipelineStateWithFunction:func_strided error:&error];
    if (!g_pipeline_l2_strided) {
      const char *msg = error ? [[error localizedDescription] UTF8String] : "pipeline strided failed";
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "pipeline_strided: %.200s", msg ? msg : "(nil)");
      g_metal_init_failed = true;
      return false;
    }

    id<MTLFunction> func_strided_i32 = [g_library newFunctionWithName:@"metal_l2_sqr_strided_i32"];
    if (func_strided_i32) {
      g_pipeline_l2_strided_i32 = [g_device newComputePipelineStateWithFunction:func_strided_i32 error:&error];
    }

    id<MTLFunction> func_batch = [g_library newFunctionWithName:@"metal_l2_sqr_batch"];
    if (func_batch) {
      g_pipeline_l2_batch = [g_device newComputePipelineStateWithFunction:func_batch error:&error];
    }

    g_command_queue = [g_device newCommandQueue];
    if (!g_command_queue) {
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "newCommandQueue nil");
      g_metal_init_failed = true;
      return false;
    }

    g_metal_initialized = true;
    return true;
  }
}

int metal_l2_query_vids(const float *query,
                        const float *vectors,
                        int64_t count,
                        int64_t dim,
                        float *distances_out)
{
  if (query == nullptr || vectors == nullptr || distances_out == nullptr) return -1;
  if (count <= 0 || dim <= 0) return -1;
  if (!init_metal()) return -1;

  @autoreleasepool {
    uint32_t u_dim = static_cast<uint32_t>(dim);
    uint32_t u_count = static_cast<uint32_t>(count);

    size_t query_bytes = static_cast<size_t>(dim) * sizeof(float);
    size_t vectors_bytes = static_cast<size_t>(count) * static_cast<size_t>(dim) * sizeof(float);
    size_t distances_bytes = static_cast<size_t>(count) * sizeof(float);

    id<MTLBuffer> query_buf = nil;
    if (is_page_aligned(query)) {
      query_buf = [g_device newBufferWithBytesNoCopy:const_cast<float *>(query)
          length:query_bytes options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      query_buf = [g_device newBufferWithBytes:query length:query_bytes
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> vectors_buf = nil;
    if (is_page_aligned(vectors)) {
      vectors_buf = [g_device newBufferWithBytesNoCopy:const_cast<float *>(vectors)
          length:vectors_bytes options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      vectors_buf = [g_device newBufferWithBytes:vectors length:vectors_bytes
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> distances_buf = nil;
    bool out_is_shared = is_page_aligned(distances_out);
    if (out_is_shared) {
      distances_buf = [g_device newBufferWithBytesNoCopy:distances_out
          length:distances_bytes
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      distances_buf = [g_device newBufferWithLength:distances_bytes
          options:MTLResourceStorageModeShared];
    }
    if (!query_buf || !vectors_buf || !distances_buf) return -1;

    id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_l2];
    [encoder setBuffer:query_buf offset:0 atIndex:0];
    [encoder setBuffer:vectors_buf offset:0 atIndex:1];
    [encoder setBuffer:distances_buf offset:0 atIndex:2];
    [encoder setBytes:&u_dim length:sizeof(uint32_t) atIndex:3];

    NSUInteger tg = g_pipeline_l2.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    MTLSize grid_size = MTLSizeMake(u_count, 1, 1);
    MTLSize group_size = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    if (!out_is_shared) {
      memcpy(distances_out, [distances_buf contents], distances_bytes);
    }
  }
  return 0;
}

int metal_l2_batch_query_vids(const float *queries,
                              const float *vectors,
                              int64_t num_queries,
                              int64_t num_vectors,
                              int64_t dim,
                              float *distances_out)
{
  if (queries == nullptr || vectors == nullptr || distances_out == nullptr) return -1;
  if (num_queries <= 0 || num_vectors <= 0 || dim <= 0) return -1;
  if (!init_metal() || !g_pipeline_l2_batch) return -1;

  @autoreleasepool {
    uint32_t u_dim = static_cast<uint32_t>(dim);
    uint32_t u_num_vectors = static_cast<uint32_t>(num_vectors);
    uint32_t total = static_cast<uint32_t>(num_queries * num_vectors);

    size_t queries_bytes = static_cast<size_t>(num_queries) * static_cast<size_t>(dim) * sizeof(float);
    size_t vectors_bytes = static_cast<size_t>(num_vectors) * static_cast<size_t>(dim) * sizeof(float);
    size_t distances_bytes = static_cast<size_t>(num_queries) * static_cast<size_t>(num_vectors) * sizeof(float);

    id<MTLBuffer> queries_buf = [g_device newBufferWithBytes:queries length:queries_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> vectors_buf = nil;
    if (is_page_aligned(vectors)) {
      vectors_buf = [g_device newBufferWithBytesNoCopy:const_cast<float *>(vectors)
          length:vectors_bytes options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      vectors_buf = [g_device newBufferWithBytes:vectors length:vectors_bytes
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> distances_buf = [g_device newBufferWithLength:distances_bytes
        options:MTLResourceStorageModeShared];
    if (!queries_buf || !vectors_buf || !distances_buf) return -1;

    id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_l2_batch];
    [encoder setBuffer:queries_buf offset:0 atIndex:0];
    [encoder setBuffer:vectors_buf offset:0 atIndex:1];
    [encoder setBuffer:distances_buf offset:0 atIndex:2];
    [encoder setBytes:&u_dim length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&u_num_vectors length:sizeof(uint32_t) atIndex:4];

    NSUInteger tg = g_pipeline_l2_batch.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    MTLSize grid_size = MTLSizeMake(total, 1, 1);
    MTLSize group_size = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    memcpy(distances_out, [distances_buf contents], distances_bytes);
  }
  return 0;
}

int metal_l2_query_vids_strided(const float *query,
                                const char *vectors_data,
                                size_t vectors_data_byte_len,
                                const uint32_t *byte_offsets,
                                int64_t count,
                                int64_t dim,
                                float *distances_out)
{
  if (query == nullptr || vectors_data == nullptr || byte_offsets == nullptr || distances_out == nullptr) return -1;
  if (count <= 0 || dim <= 0 || vectors_data_byte_len == 0) return -1;
  if (!init_metal() || !g_pipeline_l2_strided) return -1;

  @autoreleasepool {
    uint32_t u_dim = static_cast<uint32_t>(dim);
    uint32_t u_count = static_cast<uint32_t>(count);

    size_t query_bytes = static_cast<size_t>(dim) * sizeof(float);
    size_t offsets_bytes = static_cast<size_t>(count) * sizeof(uint32_t);
    size_t distances_bytes = static_cast<size_t>(count) * sizeof(float);

    id<MTLBuffer> query_buf = nil;
    if (is_page_aligned(query)) {
      query_buf = [g_device newBufferWithBytesNoCopy:const_cast<float *>(query)
          length:query_bytes
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      query_buf = [g_device newBufferWithBytes:query length:query_bytes
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> vectors_buf = nil;
    if (is_page_aligned(vectors_data)) {
      vectors_buf = [g_device newBufferWithBytesNoCopy:const_cast<char *>(vectors_data)
          length:vectors_data_byte_len
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      vectors_buf = [g_device newBufferWithBytes:vectors_data length:vectors_data_byte_len
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> offsets_buf = [g_device newBufferWithBytes:byte_offsets length:offsets_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> distances_buf = nil;
    bool out_is_shared = is_page_aligned(distances_out);
    if (out_is_shared) {
      distances_buf = [g_device newBufferWithBytesNoCopy:distances_out
          length:distances_bytes
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      distances_buf = [g_device newBufferWithLength:distances_bytes
          options:MTLResourceStorageModeShared];
    }
    if (!query_buf || !vectors_buf || !offsets_buf || !distances_buf) return -1;

    id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_l2_strided];
    [encoder setBuffer:query_buf offset:0 atIndex:0];
    [encoder setBuffer:vectors_buf offset:0 atIndex:1];
    [encoder setBuffer:offsets_buf offset:0 atIndex:2];
    [encoder setBuffer:distances_buf offset:0 atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(uint32_t) atIndex:4];

    NSUInteger tg = g_pipeline_l2_strided.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    MTLSize grid_size = MTLSizeMake(u_count, 1, 1);
    MTLSize group_size = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    if (!out_is_shared) {
      memcpy(distances_out, [distances_buf contents], distances_bytes);
    }
  }
  return 0;
}

int metal_l2_query_vids_strided_f64(const double *query,
                                   const char *vectors_data,
                                   size_t vectors_data_byte_len,
                                   const uint32_t *byte_offsets,
                                   int64_t count,
                                   int64_t dim,
                                   float *distances_out)
{
  if (query == nullptr || vectors_data == nullptr || byte_offsets == nullptr || distances_out == nullptr) return -1;
  if (count <= 0 || dim <= 0 || vectors_data_byte_len == 0) return -1;
  if (!init_metal() || !g_pipeline_l2_strided) return -1;

  // Metal does not support double; convert double -> float on CPU and use float kernel.
  const size_t num_doubles = vectors_data_byte_len / sizeof(double);
  const size_t float_data_byte_len = num_doubles * sizeof(float);
  std::vector<float> query_f(static_cast<size_t>(dim));
  std::vector<float> vectors_f(num_doubles);
  std::vector<uint32_t> byte_offsets_f(static_cast<size_t>(count));

  for (int64_t i = 0; i < dim; i++) {
    query_f[static_cast<size_t>(i)] = static_cast<float>(query[i]);
  }
  const double *vec_d = reinterpret_cast<const double *>(vectors_data);
  for (size_t i = 0; i < num_doubles; i++) {
    vectors_f[i] = static_cast<float>(vec_d[i]);
  }
  for (int64_t i = 0; i < count; i++) {
    byte_offsets_f[static_cast<size_t>(i)] = static_cast<uint32_t>(byte_offsets[i] / 2);
  }

  return metal_l2_query_vids_strided(
      query_f.data(),
      reinterpret_cast<const char *>(vectors_f.data()),
      float_data_byte_len,
      byte_offsets_f.data(),
      count,
      dim,
      distances_out);
}

int metal_l2_query_vids_strided_i32(const int32_t *query,
                                    const char *vectors_data,
                                    size_t vectors_data_byte_len,
                                    const uint32_t *byte_offsets,
                                    int64_t count,
                                    int64_t dim,
                                    float *distances_out)
{
  if (query == nullptr || vectors_data == nullptr || byte_offsets == nullptr || distances_out == nullptr) return -1;
  if (count <= 0 || dim <= 0 || vectors_data_byte_len == 0) return -1;
  if (!init_metal() || !g_pipeline_l2_strided_i32) return -1;

  @autoreleasepool {
    uint32_t u_dim = static_cast<uint32_t>(dim);
    uint32_t u_count = static_cast<uint32_t>(count);

    size_t query_bytes = static_cast<size_t>(dim) * sizeof(int32_t);
    size_t offsets_bytes = static_cast<size_t>(count) * sizeof(uint32_t);
    size_t distances_bytes = static_cast<size_t>(count) * sizeof(float);

    id<MTLBuffer> query_buf = [g_device newBufferWithBytes:query length:query_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> vectors_buf = [g_device newBufferWithBytes:vectors_data length:vectors_data_byte_len
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> offsets_buf = [g_device newBufferWithBytes:byte_offsets length:offsets_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> distances_buf = [g_device newBufferWithLength:distances_bytes
        options:MTLResourceStorageModeShared];
    if (!query_buf || !vectors_buf || !offsets_buf || !distances_buf) return -1;

    id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_l2_strided_i32];
    [encoder setBuffer:query_buf offset:0 atIndex:0];
    [encoder setBuffer:vectors_buf offset:0 atIndex:1];
    [encoder setBuffer:offsets_buf offset:0 atIndex:2];
    [encoder setBuffer:distances_buf offset:0 atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(uint32_t) atIndex:4];

    NSUInteger tg = g_pipeline_l2_strided_i32.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    MTLSize grid_size = MTLSizeMake(u_count, 1, 1);
    MTLSize group_size = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    memcpy(distances_out, [distances_buf contents], distances_bytes);
  }
  return 0;
}

bool is_metal_ready()
{
  return init_metal();
}

const char *get_metal_l2_init_error()
{
  if (g_metal_initialized) return "";
  if (g_metal_init_failed) return g_metal_init_error;
  init_metal();
  return g_metal_init_failed ? g_metal_init_error : "";
}

}  // namespace vector_metal
}  // namespace gpu_acc
}  // namespace oceanbase

#else  // !__APPLE__

namespace oceanbase
{
namespace gpu_acc
{
namespace vector_metal
{

int metal_l2_query_vids(const float * /*query*/,
                        const float * /*vectors*/,
                        int64_t /*count*/,
                        int64_t /*dim*/,
                        float * /*distances_out*/)
{
  return -1;
}

int metal_l2_query_vids_strided(const float * /*query*/,
                                const char * /*vectors_data*/,
                                size_t /*vectors_data_byte_len*/,
                                const uint32_t * /*byte_offsets*/,
                                int64_t /*count*/,
                                int64_t /*dim*/,
                                float * /*distances_out*/)
{
  return -1;
}

int metal_l2_query_vids_strided_f64(const double * /*query*/,
                                    const char * /*vectors_data*/,
                                    size_t /*vectors_data_byte_len*/,
                                    const uint32_t * /*byte_offsets*/,
                                    int64_t /*count*/,
                                    int64_t /*dim*/,
                                    float * /*distances_out*/)
{
  return -1;
}

int metal_l2_query_vids_strided_i32(const int32_t * /*query*/,
                                    const char * /*vectors_data*/,
                                    size_t /*vectors_data_byte_len*/,
                                    const uint32_t * /*byte_offsets*/,
                                    int64_t /*count*/,
                                    int64_t /*dim*/,
                                    float * /*distances_out*/)
{
  return -1;
}

int metal_l2_batch_query_vids(const float * /*queries*/,
                              const float * /*vectors*/,
                              int64_t /*num_queries*/,
                              int64_t /*num_vectors*/,
                              int64_t /*dim*/,
                              float * /*distances_out*/)
{
  return -1;
}

bool is_metal_ready()
{
  return false;
}

const char *get_metal_l2_init_error()
{
  return "Metal only on Mac";
}

}  // namespace vector_metal
}  // namespace gpu_acc
}  // namespace oceanbase

#endif  // __APPLE__
