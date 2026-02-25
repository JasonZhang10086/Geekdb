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

#include "gpu_acc/metal/vector/ob_metal_cosine.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
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

static id<MTLDevice> g_cosine_device = nil;
static id<MTLLibrary> g_cosine_library = nil;
static id<MTLCommandQueue> g_cosine_command_queue = nil;
static id<MTLComputePipelineState> g_pipeline_cosine = nil;
static id<MTLComputePipelineState> g_pipeline_cosine_strided = nil;
static id<MTLComputePipelineState> g_pipeline_cosine_strided_f64 = nil;
static bool g_cosine_metal_initialized = false;
static bool g_cosine_metal_init_failed = false;
static char g_cosine_metal_init_error[256] = {0};

static bool cosine_is_page_aligned(const void *ptr)
{
  return (reinterpret_cast<uintptr_t>(ptr) & (METAL_PAGE_SIZE - 1)) == 0;
}

static float compute_norm_float(const float *vec, int64_t dim)
{
  double sum = 0.0;
  for (int64_t j = 0; j < dim; j++) {
    double v = static_cast<double>(vec[j]);
    sum += v * v;
  }
  return static_cast<float>(std::sqrt(sum));
}

static float compute_norm_double(const double *vec, int64_t dim)
{
  double sum = 0.0;
  for (int64_t j = 0; j < dim; j++) {
    sum += vec[j] * vec[j];
  }
  return static_cast<float>(std::sqrt(sum));
}

static bool init_cosine_metal()
{
  if (g_cosine_metal_initialized) return true;
  if (g_cosine_metal_init_failed) return false;

  @autoreleasepool {
    g_cosine_device = MTLCreateSystemDefaultDevice();
    if (!g_cosine_device) {
      NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
      if (devices.count > 0) {
        g_cosine_device = devices[0];
      }
    }
    if (!g_cosine_device) {
      snprintf(g_cosine_metal_init_error, sizeof(g_cosine_metal_init_error), "no Metal device");
      g_cosine_metal_init_failed = true;
      return false;
    }

    NSError *error = nil;
    NSString *source = @R"metal(
#include <metal_stdlib>
using namespace metal;

// cosine_distance = 1 - dot(q,v) / (norm_q * norm_v); norm_q passed as constant
kernel void metal_cosine(
    device const float *query   [[buffer(0)]],
    device const float *vectors [[buffer(1)]],
    device float *distances     [[buffer(2)]],
    constant uint &dim         [[buffer(3)]],
    constant float &norm_query [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint d = dim;
  device const float *v = vectors + gid * d;
  float dot = 0.0f;
  float norm_v_sq = 0.0f;
  for (uint j = 0; j < d; j++) {
    dot += query[j] * v[j];
    norm_v_sq += v[j] * v[j];
  }
  float norm_v = sqrt(norm_v_sq);
  float norm_q = norm_query;
  if (norm_q < 1e-10f || norm_v < 1e-10f) {
    distances[gid] = 0.0f;
  } else {
    float sim = dot / (norm_q * norm_v);
    if (sim > 1.0f) sim = 1.0f;
    else if (sim < -1.0f) sim = -1.0f;
    distances[gid] = 1.0f - sim;
  }
}

kernel void metal_cosine_strided(
    device const float *query       [[buffer(0)]],
    device const char *vectors_data  [[buffer(1)]],
    device const uint *byte_offsets  [[buffer(2)]],
    device float *distances         [[buffer(3)]],
    constant uint &dim              [[buffer(4)]],
    constant float &norm_query      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  uint d = dim;
  device const float *v = (device const float *)(vectors_data + byte_offsets[gid]);
  float dot = 0.0f;
  float norm_v_sq = 0.0f;
  for (uint j = 0; j < d; j++) {
    dot += query[j] * v[j];
    norm_v_sq += v[j] * v[j];
  }
  float norm_v = sqrt(norm_v_sq);
  float norm_q = norm_query;
  if (norm_q < 1e-10f || norm_v < 1e-10f) {
    distances[gid] = 0.0f;
  } else {
    float sim = dot / (norm_q * norm_v);
    if (sim > 1.0f) sim = 1.0f;
    else if (sim < -1.0f) sim = -1.0f;
    distances[gid] = 1.0f - sim;
  }
}

kernel void metal_cosine_strided_f64(
    device const double *query      [[buffer(0)]],
    device const char *vectors_data  [[buffer(1)]],
    device const uint *byte_offsets  [[buffer(2)]],
    device float *distances         [[buffer(3)]],
    constant uint &dim              [[buffer(4)]],
    constant float &norm_query      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  uint d = dim;
  device const double *v = (device const double *)(vectors_data + byte_offsets[gid]);
  float dot = 0.0f;
  float norm_v_sq = 0.0f;
  for (uint j = 0; j < d; j++) {
    float q = static_cast<float>(query[j]);
    float vf = static_cast<float>(v[j]);
    dot += q * vf;
    norm_v_sq += vf * vf;
  }
  float norm_v = sqrt(norm_v_sq);
  float norm_q = norm_query;
  if (norm_q < 1e-10f || norm_v < 1e-10f) {
    distances[gid] = 0.0f;
  } else {
    float sim = dot / (norm_q * norm_v);
    if (sim > 1.0f) sim = 1.0f;
    else if (sim < -1.0f) sim = -1.0f;
    distances[gid] = 1.0f - sim;
  }
}
)metal";

    g_cosine_library = [g_cosine_device newLibraryWithSource:source options:nil error:&error];
    if (!g_cosine_library) {
      const char *msg = error ? [[error localizedDescription] UTF8String] : "library compile failed";
      snprintf(g_cosine_metal_init_error, sizeof(g_cosine_metal_init_error), "library: %.200s", msg ? msg : "(nil)");
      g_cosine_metal_init_failed = true;
      return false;
    }

    id<MTLFunction> func = [g_cosine_library newFunctionWithName:@"metal_cosine"];
    if (!func) {
      snprintf(g_cosine_metal_init_error, sizeof(g_cosine_metal_init_error), "function metal_cosine not found");
      g_cosine_metal_init_failed = true;
      return false;
    }
    g_pipeline_cosine = [g_cosine_device newComputePipelineStateWithFunction:func error:&error];
    if (!g_pipeline_cosine) {
      const char *msg = error ? [[error localizedDescription] UTF8String] : "pipeline failed";
      snprintf(g_cosine_metal_init_error, sizeof(g_cosine_metal_init_error), "pipeline: %.200s", msg ? msg : "(nil)");
      g_cosine_metal_init_failed = true;
      return false;
    }

    id<MTLFunction> func_strided = [g_cosine_library newFunctionWithName:@"metal_cosine_strided"];
    if (!func_strided) {
      snprintf(g_cosine_metal_init_error, sizeof(g_cosine_metal_init_error), "function metal_cosine_strided not found");
      g_cosine_metal_init_failed = true;
      return false;
    }
    g_pipeline_cosine_strided = [g_cosine_device newComputePipelineStateWithFunction:func_strided error:&error];
    if (!g_pipeline_cosine_strided) {
      const char *msg = error ? [[error localizedDescription] UTF8String] : "pipeline strided failed";
      snprintf(g_cosine_metal_init_error, sizeof(g_cosine_metal_init_error), "pipeline_strided: %.200s", msg ? msg : "(nil)");
      g_cosine_metal_init_failed = true;
      return false;
    }

    id<MTLFunction> func_strided_f64 = [g_cosine_library newFunctionWithName:@"metal_cosine_strided_f64"];
    if (func_strided_f64) {
      g_pipeline_cosine_strided_f64 = [g_cosine_device newComputePipelineStateWithFunction:func_strided_f64 error:&error];
    }

    g_cosine_command_queue = [g_cosine_device newCommandQueue];
    if (!g_cosine_command_queue) {
      snprintf(g_cosine_metal_init_error, sizeof(g_cosine_metal_init_error), "newCommandQueue nil");
      g_cosine_metal_init_failed = true;
      return false;
    }

    g_cosine_metal_initialized = true;
    return true;
  }
}

int metal_cosine_query_vids(const float *query,
                            const float *vectors,
                            int64_t count,
                            int64_t dim,
                            float *distances_out)
{
  if (query == nullptr || vectors == nullptr || distances_out == nullptr) return -1;
  if (count <= 0 || dim <= 0) return -1;
  if (!init_cosine_metal()) return -1;

  float norm_query = compute_norm_float(query, dim);

  @autoreleasepool {
    uint32_t u_dim = static_cast<uint32_t>(dim);
    uint32_t u_count = static_cast<uint32_t>(count);

    size_t query_bytes = static_cast<size_t>(dim) * sizeof(float);
    size_t vectors_bytes = static_cast<size_t>(count) * static_cast<size_t>(dim) * sizeof(float);
    size_t distances_bytes = static_cast<size_t>(count) * sizeof(float);

    id<MTLBuffer> query_buf = nil;
    if (cosine_is_page_aligned(query)) {
      query_buf = [g_cosine_device newBufferWithBytesNoCopy:const_cast<float *>(query)
          length:query_bytes options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      query_buf = [g_cosine_device newBufferWithBytes:query length:query_bytes
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> vectors_buf = nil;
    if (cosine_is_page_aligned(vectors)) {
      vectors_buf = [g_cosine_device newBufferWithBytesNoCopy:const_cast<float *>(vectors)
          length:vectors_bytes options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      vectors_buf = [g_cosine_device newBufferWithBytes:vectors length:vectors_bytes
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> distances_buf = nil;
    bool out_is_shared = cosine_is_page_aligned(distances_out);
    if (out_is_shared) {
      distances_buf = [g_cosine_device newBufferWithBytesNoCopy:distances_out
          length:distances_bytes
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      distances_buf = [g_cosine_device newBufferWithLength:distances_bytes
          options:MTLResourceStorageModeShared];
    }
    if (!query_buf || !vectors_buf || !distances_buf) return -1;

    id<MTLCommandBuffer> cmd_buf = [g_cosine_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_cosine];
    [encoder setBuffer:query_buf offset:0 atIndex:0];
    [encoder setBuffer:vectors_buf offset:0 atIndex:1];
    [encoder setBuffer:distances_buf offset:0 atIndex:2];
    [encoder setBytes:&u_dim length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&norm_query length:sizeof(float) atIndex:4];

    NSUInteger tg = g_pipeline_cosine.maxTotalThreadsPerThreadgroup;
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

int metal_cosine_query_vids_strided(const float *query,
                                    const char *vectors_data,
                                    size_t vectors_data_byte_len,
                                    const uint32_t *byte_offsets,
                                    int64_t count,
                                    int64_t dim,
                                    float *distances_out)
{
  if (query == nullptr || vectors_data == nullptr || byte_offsets == nullptr || distances_out == nullptr) return -1;
  if (count <= 0 || dim <= 0 || vectors_data_byte_len == 0) return -1;
  if (!init_cosine_metal() || !g_pipeline_cosine_strided) return -1;

  float norm_query = compute_norm_float(query, dim);

  @autoreleasepool {
    uint32_t u_dim = static_cast<uint32_t>(dim);
    uint32_t u_count = static_cast<uint32_t>(count);

    size_t query_bytes = static_cast<size_t>(dim) * sizeof(float);
    size_t offsets_bytes = static_cast<size_t>(count) * sizeof(uint32_t);
    size_t distances_bytes = static_cast<size_t>(count) * sizeof(float);

    id<MTLBuffer> query_buf = nil;
    if (cosine_is_page_aligned(query)) {
      query_buf = [g_cosine_device newBufferWithBytesNoCopy:const_cast<float *>(query)
          length:query_bytes
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      query_buf = [g_cosine_device newBufferWithBytes:query length:query_bytes
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> vectors_buf = nil;
    if (cosine_is_page_aligned(vectors_data)) {
      vectors_buf = [g_cosine_device newBufferWithBytesNoCopy:const_cast<char *>(vectors_data)
          length:vectors_data_byte_len
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      vectors_buf = [g_cosine_device newBufferWithBytes:vectors_data length:vectors_data_byte_len
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> offsets_buf = [g_cosine_device newBufferWithBytes:byte_offsets length:offsets_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> distances_buf = nil;
    bool out_is_shared = cosine_is_page_aligned(distances_out);
    if (out_is_shared) {
      distances_buf = [g_cosine_device newBufferWithBytesNoCopy:distances_out
          length:distances_bytes
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      distances_buf = [g_cosine_device newBufferWithLength:distances_bytes
          options:MTLResourceStorageModeShared];
    }
    if (!query_buf || !vectors_buf || !offsets_buf || !distances_buf) return -1;

    id<MTLCommandBuffer> cmd_buf = [g_cosine_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_cosine_strided];
    [encoder setBuffer:query_buf offset:0 atIndex:0];
    [encoder setBuffer:vectors_buf offset:0 atIndex:1];
    [encoder setBuffer:offsets_buf offset:0 atIndex:2];
    [encoder setBuffer:distances_buf offset:0 atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&norm_query length:sizeof(float) atIndex:5];

    NSUInteger tg = g_pipeline_cosine_strided.maxTotalThreadsPerThreadgroup;
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

int metal_cosine_query_vids_strided_f64(const double *query,
                                         const char *vectors_data,
                                         size_t vectors_data_byte_len,
                                         const uint32_t *byte_offsets,
                                         int64_t count,
                                         int64_t dim,
                                         float *distances_out)
{
  if (query == nullptr || vectors_data == nullptr || byte_offsets == nullptr || distances_out == nullptr) return -1;
  if (count <= 0 || dim <= 0 || vectors_data_byte_len == 0) return -1;
  if (!init_cosine_metal() || !g_pipeline_cosine_strided_f64) return -1;

  float norm_query = compute_norm_double(query, dim);

  @autoreleasepool {
    uint32_t u_dim = static_cast<uint32_t>(dim);
    uint32_t u_count = static_cast<uint32_t>(count);

    size_t query_bytes = static_cast<size_t>(dim) * sizeof(double);
    size_t offsets_bytes = static_cast<size_t>(count) * sizeof(uint32_t);
    size_t distances_bytes = static_cast<size_t>(count) * sizeof(float);

    id<MTLBuffer> query_buf = [g_cosine_device newBufferWithBytes:query length:query_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> vectors_buf = nil;
    if (cosine_is_page_aligned(vectors_data)) {
      vectors_buf = [g_cosine_device newBufferWithBytesNoCopy:const_cast<char *>(vectors_data)
          length:vectors_data_byte_len
          options:MTLResourceStorageModeShared
          deallocator:^(void * /*ptr*/, NSUInteger /*len*/) {}];
    } else {
      vectors_buf = [g_cosine_device newBufferWithBytes:vectors_data length:vectors_data_byte_len
          options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> offsets_buf = [g_cosine_device newBufferWithBytes:byte_offsets length:offsets_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> distances_buf = [g_cosine_device newBufferWithLength:distances_bytes
        options:MTLResourceStorageModeShared];
    if (!query_buf || !vectors_buf || !offsets_buf || !distances_buf) return -1;

    id<MTLCommandBuffer> cmd_buf = [g_cosine_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_cosine_strided_f64];
    [encoder setBuffer:query_buf offset:0 atIndex:0];
    [encoder setBuffer:vectors_buf offset:0 atIndex:1];
    [encoder setBuffer:offsets_buf offset:0 atIndex:2];
    [encoder setBuffer:distances_buf offset:0 atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&norm_query length:sizeof(float) atIndex:5];

    NSUInteger tg = g_pipeline_cosine_strided_f64.maxTotalThreadsPerThreadgroup;
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

bool is_metal_cosine_ready()
{
  return init_cosine_metal();
}

const char *get_metal_cosine_init_error()
{
  if (g_cosine_metal_initialized) return "";
  if (g_cosine_metal_init_failed) return g_cosine_metal_init_error;
  init_cosine_metal();
  return g_cosine_metal_init_failed ? g_cosine_metal_init_error : "";
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

int metal_cosine_query_vids(const float * /*query*/,
                            const float * /*vectors*/,
                            int64_t /*count*/,
                            int64_t /*dim*/,
                            float * /*distances_out*/)
{
  return -1;
}

int metal_cosine_query_vids_strided(const float * /*query*/,
                                    const char * /*vectors_data*/,
                                    size_t /*vectors_data_byte_len*/,
                                    const uint32_t * /*byte_offsets*/,
                                    int64_t /*count*/,
                                    int64_t /*dim*/,
                                    float * /*distances_out*/)
{
  return -1;
}

int metal_cosine_query_vids_strided_f64(const double * /*query*/,
                                         const char * /*vectors_data*/,
                                         size_t /*vectors_data_byte_len*/,
                                         const uint32_t * /*byte_offsets*/,
                                         int64_t /*count*/,
                                         int64_t /*dim*/,
                                         float * /*distances_out*/)
{
  return -1;
}

bool is_metal_cosine_ready()
{
  return false;
}

const char *get_metal_cosine_init_error()
{
  return "Metal only on Mac";
}

}  // namespace vector_metal
}  // namespace gpu_acc
}  // namespace oceanbase

#endif  // __APPLE__
