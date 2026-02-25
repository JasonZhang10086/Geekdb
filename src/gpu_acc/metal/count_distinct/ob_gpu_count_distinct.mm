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

#include "gpu_acc/metal/count_distinct/ob_gpu_count_distinct.h"
#include "gpu_acc/ob_gpu_config.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <new>

namespace oceanbase
{
namespace gpu_acc
{

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_command_queue = nil;

static id<MTLComputePipelineState> g_pipeline_int64 = nil;
static id<MTLComputePipelineState> g_pipeline_double = nil;
static id<MTLComputePipelineState> g_pipeline_string = nil;

static bool g_metal_initialized = false;
static bool g_metal_init_failed = false;
static char g_metal_init_error[256] = {0};

static constexpr int64_t GPU_BATCH_SIZE = 16 * 1024 * 1024; // 16M rows per batch
static constexpr int64_t GPU_MAX_TABLE_ENTRIES = 1LL << 26; // 64M entries max per table

static bool init_metal()
{
  if (g_metal_initialized) return true;
  if (g_metal_init_failed) return false;

  @autoreleasepool {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
      // Daemon/CLI processes often get nil from MTLCreateSystemDefaultDevice(); use any available device.
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

    // Embedded shader source
    NSString *source = @R"metal(
#include <metal_stdlib>
using namespace metal;

// Lock-free GPU hash table: stores hash fingerprints via atomic CAS.
// No spin-waits, no state machines, guaranteed bounded execution.

// MurmurHash3 64-bit finalizer → split into (index_hash, cmp_hash)
inline uint2 murmur_hash64(int64_t val) {
  uint64_t h = (uint64_t)val;
  h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return uint2((uint)(h & 0xFFFFFFFF), (uint)(h >> 32));
}

// FNV-1a 64-bit for strings → split into (index_hash, cmp_hash)
inline uint2 fnv1a_hash64(device const char *data, uint offset, uint len, uint max_len) {
  uint64_t h = 14695981039346656037ULL;
  for (uint i = 0; i < len && i < max_len; i++) {
    h ^= (uint64_t)(uint8_t)data[offset + i];
    h *= 1099511628211ULL;
  }
  return uint2((uint)(h & 0xFFFFFFFF), (uint)(h >> 32));
}

// INT64: dual-hash lock-free CAS (index from lower 32, fingerprint from upper 32)
kernel void count_distinct_int64(
    device const int64_t *input [[buffer(0)]],
    device atomic_uint *hash_table [[buffer(1)]],
    device int64_t *key_table [[buffer(2)]],
    device atomic_uint *distinct_count [[buffer(3)]],
    constant uint &num_elements [[buffer(4)]],
    constant uint &table_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= num_elements) return;
  int64_t val = input[gid];
  uint2 hh = murmur_hash64(val);
  uint h_idx = hh.x;
  uint h_cmp = hh.y | 1u;
  uint idx = h_idx % table_size;
  for (uint probe = 0; probe < table_size; probe++) {
    uint slot_idx = (idx + probe) % table_size;
    uint expected = 0;
    if (atomic_compare_exchange_weak_explicit(&hash_table[slot_idx], &expected, h_cmp,
        memory_order_relaxed, memory_order_relaxed)) {
      key_table[slot_idx] = val;
      atomic_fetch_add_explicit(distinct_count, 1u, memory_order_relaxed);
      return;
    }
    if (expected == h_cmp) return;
  }
}

// DOUBLE: 64-bit precision via bit-pattern hashing (treats double bits as int64_t)
kernel void count_distinct_double64(
    device const int64_t *input [[buffer(0)]],
    device atomic_uint *hash_table [[buffer(1)]],
    device int64_t *key_table [[buffer(2)]],
    device atomic_uint *distinct_count [[buffer(3)]],
    constant uint &num_elements [[buffer(4)]],
    constant uint &table_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= num_elements) return;
  int64_t val = input[gid];
  // normalize -0.0 to +0.0
  if ((uint64_t)val == 0x8000000000000000ULL) val = 0;
  uint2 hh = murmur_hash64(val);
  uint h_idx = hh.x;
  uint h_cmp = hh.y | 1u;
  uint idx = h_idx % table_size;
  for (uint probe = 0; probe < table_size; probe++) {
    uint slot_idx = (idx + probe) % table_size;
    uint expected = 0;
    if (atomic_compare_exchange_weak_explicit(&hash_table[slot_idx], &expected, h_cmp,
        memory_order_relaxed, memory_order_relaxed)) {
      key_table[slot_idx] = val;
      atomic_fetch_add_explicit(distinct_count, 1u, memory_order_relaxed);
      return;
    }
    if (expected == h_cmp) return;
  }
}

struct StringHeader { uint offset; uint length; };

// STRING: dual-hash lock-free CAS (64-bit FNV-1a split into index + fingerprint)
kernel void count_distinct_string(
    device const char *str_data [[buffer(0)]],
    device const StringHeader *str_headers [[buffer(1)]],
    device atomic_uint *hash_table [[buffer(2)]],
    device uint *hash_key_indices [[buffer(3)]],
    device atomic_uint *distinct_count [[buffer(4)]],
    constant uint &num_elements [[buffer(5)]],
    constant uint &table_size [[buffer(6)]],
    constant uint &max_str_len [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= num_elements) return;
  uint str_offset = str_headers[gid].offset;
  uint str_len = str_headers[gid].length;
  uint2 hh = fnv1a_hash64(str_data, str_offset, str_len, max_str_len);
  uint h_idx = hh.x;
  uint h_cmp = hh.y | 1u;
  uint idx = h_idx % table_size;
  for (uint probe = 0; probe < table_size; probe++) {
    uint slot_idx = (idx + probe) % table_size;
    uint expected = 0;
    if (atomic_compare_exchange_weak_explicit(&hash_table[slot_idx], &expected, h_cmp,
        memory_order_relaxed, memory_order_relaxed)) {
      hash_key_indices[slot_idx] = gid;
      atomic_fetch_add_explicit(distinct_count, 1u, memory_order_relaxed);
      return;
    }
    if (expected == h_cmp) return;
  }
}
)metal";

    // Use default compile options (nil) for maximum compatibility.
    g_library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!g_library) {
      const char *msg = error ? [[error localizedDescription] UTF8String] : "library compile failed";
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "library: %.200s", msg ? msg : "(nil)");
      g_metal_init_failed = true;
      return false;
    }

    auto create_pipeline = [&](const char *name) -> id<MTLComputePipelineState> {
      id<MTLFunction> func = [g_library newFunctionWithName:
          [NSString stringWithUTF8String:name]];
      if (!func) return nil;
      return [g_device newComputePipelineStateWithFunction:func error:&error];
    };

    g_pipeline_int64 = create_pipeline("count_distinct_int64");
    g_pipeline_double = create_pipeline("count_distinct_double64");
    g_pipeline_string = create_pipeline("count_distinct_string");

    if (!g_pipeline_int64 || !g_pipeline_double || !g_pipeline_string) {
      const char *msg = error ? [[error localizedDescription] UTF8String] : "pipeline failed";
      snprintf(g_metal_init_error, sizeof(g_metal_init_error), "pipeline: %.200s", msg ? msg : "(nil)");
      g_metal_init_failed = true;
      return false;
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

static uint32_t calc_table_size(int64_t estimated_distinct)
{
  uint64_t size = (uint64_t)(estimated_distinct * 100 / GPU_HASH_TABLE_LOAD_FACTOR_PCT);
  uint64_t power = 1;
  while (power < size) power <<= 1;
  if (power > (uint64_t)GPU_MAX_TABLE_ENTRIES) power = (uint64_t)GPU_MAX_TABLE_ENTRIES;
  if (power < 1024) power = 1024;
  return (uint32_t)power;
}

// macOS Metal max buffer size is 256MB; larger allocations fail.
static constexpr size_t MAX_INPUT_BUFFER_BYTES = 256 * 1024 * 1024;

/*
 * Streaming batch GPU count distinct for int64.
 * Uses single input buffer when data fits in maxBufferLength, else per-batch buffers.
 */
int gpu_count_distinct_int64(const int64_t *data, int64_t num_rows,
                             GpuCountDistinctResult &result)
{
  if (!init_metal()) return -1;
  if (num_rows <= 0) {
    result.distinct_count = 0;
    result.used_gpu = true;
    return 0;
  }
  if (data == nullptr) return -1;

  @autoreleasepool {
    auto start = std::chrono::high_resolution_clock::now();

    size_t max_buf = [g_device maxBufferLength];
    if (max_buf > MAX_INPUT_BUFFER_BYTES) max_buf = MAX_INPUT_BUFFER_BYTES;

    int64_t estimated_distinct = std::min(num_rows, (int64_t)GPU_MAX_TABLE_ENTRIES);
    uint32_t table_size = calc_table_size(estimated_distinct);

    size_t total_input_bytes = (size_t)num_rows * sizeof(int64_t);
    id<MTLBuffer> input_buf = nil;
    if (total_input_bytes <= max_buf) {
      input_buf = [g_device newBufferWithBytes:data
          length:total_input_bytes options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> hash_table_buf = [g_device newBufferWithLength:table_size * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_table_buf = [g_device newBufferWithLength:table_size * sizeof(int64_t)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> count_buf = [g_device newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared];

    if (!hash_table_buf || !key_table_buf || !count_buf) return -1;
    if (total_input_bytes <= max_buf && !input_buf) return -1;

    memset([hash_table_buf contents], 0, table_size * sizeof(uint32_t));
    memset([count_buf contents], 0, sizeof(uint32_t));

    NSUInteger thread_group_size = g_pipeline_int64.maxTotalThreadsPerThreadgroup;
    if (thread_group_size > 256) thread_group_size = 256;

    id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_int64];
    [encoder setBuffer:hash_table_buf offset:0 atIndex:1];
    [encoder setBuffer:key_table_buf offset:0 atIndex:2];
    [encoder setBuffer:count_buf offset:0 atIndex:3];
    [encoder setBytes:&table_size length:sizeof(uint32_t) atIndex:5];

    for (int64_t offset = 0; offset < num_rows; offset += GPU_BATCH_SIZE) {
      int64_t batch_rows = std::min(GPU_BATCH_SIZE, num_rows - offset);
      uint32_t batch_elements = (uint32_t)batch_rows;
      size_t batch_bytes = (size_t)batch_rows * sizeof(int64_t);

      if (input_buf) {
        [encoder setBuffer:input_buf offset:offset * sizeof(int64_t) atIndex:0];
      } else {
        id<MTLBuffer> batch_buf = [g_device newBufferWithBytes:(data + offset)
            length:batch_bytes options:MTLResourceStorageModeShared];
        if (!batch_buf) return -1;
        [encoder setBuffer:batch_buf offset:0 atIndex:0];
      }
      [encoder setBytes:&batch_elements length:sizeof(uint32_t) atIndex:4];

      MTLSize grid_size = MTLSizeMake(batch_elements, 1, 1);
      MTLSize group_size = MTLSizeMake(thread_group_size, 1, 1);
      [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];

      if (offset + GPU_BATCH_SIZE < num_rows) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    }

    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    uint32_t *count_ptr = (uint32_t *)[count_buf contents];
    auto end = std::chrono::high_resolution_clock::now();

    result.distinct_count = *count_ptr;
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.used_gpu = true;
    return 0;
  }
}

int gpu_count_distinct_double(const double *data, int64_t num_rows,
                              GpuCountDistinctResult &result)
{
  if (!init_metal()) return -1;
  if (num_rows <= 0) {
    result.distinct_count = 0;
    result.used_gpu = true;
    return 0;
  }
  if (data == nullptr) return -1;

  @autoreleasepool {
    auto start = std::chrono::high_resolution_clock::now();

    size_t max_buf = [g_device maxBufferLength];
    if (max_buf > MAX_INPUT_BUFFER_BYTES) max_buf = MAX_INPUT_BUFFER_BYTES;

    int64_t estimated_distinct = std::min(num_rows, (int64_t)GPU_MAX_TABLE_ENTRIES);
    uint32_t table_size = calc_table_size(estimated_distinct);

    size_t total_input_bytes = (size_t)num_rows * sizeof(int64_t);
    id<MTLBuffer> input_buf = nil;
    const int64_t *data_as_i64 = reinterpret_cast<const int64_t *>(data);
    if (total_input_bytes <= max_buf) {
      input_buf = [g_device newBufferWithBytes:data_as_i64
          length:total_input_bytes options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> hash_table_buf = [g_device newBufferWithLength:table_size * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_table_buf = [g_device newBufferWithLength:table_size * sizeof(int64_t)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> count_buf = [g_device newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared];

    if (!hash_table_buf || !key_table_buf || !count_buf) return -1;
    if (total_input_bytes <= max_buf && !input_buf) return -1;

    memset([hash_table_buf contents], 0, table_size * sizeof(uint32_t));
    memset([count_buf contents], 0, sizeof(uint32_t));

    NSUInteger thread_group_size = g_pipeline_double.maxTotalThreadsPerThreadgroup;
    if (thread_group_size > 256) thread_group_size = 256;

    id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_double];
    [encoder setBuffer:hash_table_buf offset:0 atIndex:1];
    [encoder setBuffer:key_table_buf offset:0 atIndex:2];
    [encoder setBuffer:count_buf offset:0 atIndex:3];
    [encoder setBytes:&table_size length:sizeof(uint32_t) atIndex:5];

    for (int64_t offset = 0; offset < num_rows; offset += GPU_BATCH_SIZE) {
      int64_t batch_rows = std::min(GPU_BATCH_SIZE, num_rows - offset);
      uint32_t batch_elements = (uint32_t)batch_rows;
      size_t batch_bytes = (size_t)batch_rows * sizeof(int64_t);

      if (input_buf) {
        [encoder setBuffer:input_buf offset:offset * sizeof(int64_t) atIndex:0];
      } else {
        id<MTLBuffer> batch_buf = [g_device newBufferWithBytes:(data_as_i64 + offset)
            length:batch_bytes options:MTLResourceStorageModeShared];
        if (!batch_buf) return -1;
        [encoder setBuffer:batch_buf offset:0 atIndex:0];
      }
      [encoder setBytes:&batch_elements length:sizeof(uint32_t) atIndex:4];

      MTLSize grid_size = MTLSizeMake(batch_elements, 1, 1);
      MTLSize group_size = MTLSizeMake(thread_group_size, 1, 1);
      [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];

      if (offset + GPU_BATCH_SIZE < num_rows) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    }

    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    uint32_t *count_ptr = (uint32_t *)[count_buf contents];
    auto end = std::chrono::high_resolution_clock::now();

    result.distinct_count = *count_ptr;
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.used_gpu = true;
    return 0;
  }
}

int gpu_count_distinct_string(const GpuStringData &str_data, int64_t num_rows,
                              GpuCountDistinctResult &result)
{
  if (!init_metal()) return -1;
  if (num_rows <= 0) {
    result.distinct_count = 0;
    result.used_gpu = true;
    return 0;
  }

  @autoreleasepool {
    auto start = std::chrono::high_resolution_clock::now();

    int64_t estimated_distinct = std::min(num_rows, (int64_t)GPU_MAX_TABLE_ENTRIES);
    uint32_t table_size = calc_table_size(estimated_distinct);
    uint32_t max_str_len = str_data.max_str_len ? str_data.max_str_len : 1;

    uint64_t total_str_bytes = 0;
    if (num_rows > 0) {
      total_str_bytes = str_data.offsets[num_rows - 1] + str_data.lengths[num_rows - 1];
    }
    const void *str_src = str_data.flat_buffer;
    static const char dummy_byte = 0;
    if (total_str_bytes == 0) {
      total_str_bytes = 1;
      str_src = &dummy_byte;
    }

    struct StringHeader { uint32_t offset; uint32_t length; };

    StringHeader *all_headers = new (std::nothrow) StringHeader[num_rows];
    if (!all_headers) return -1;
    for (int64_t i = 0; i < num_rows; i++) {
      all_headers[i].offset = str_data.offsets[i];
      all_headers[i].length = str_data.lengths[i];
    }

    id<MTLBuffer> str_buf = [g_device newBufferWithBytes:str_src
        length:(NSUInteger)total_str_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdr_buf = [g_device newBufferWithBytes:all_headers
        length:(NSUInteger)(num_rows * sizeof(StringHeader)) options:MTLResourceStorageModeShared];
    delete[] all_headers;

    id<MTLBuffer> hash_table_buf = [g_device newBufferWithLength:table_size * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_idx_buf = [g_device newBufferWithLength:table_size * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> count_buf = [g_device newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared];

    if (!str_buf || !hdr_buf || !hash_table_buf || !key_idx_buf || !count_buf) return -1;

    memset([hash_table_buf contents], 0, table_size * sizeof(uint32_t));
    memset([count_buf contents], 0, sizeof(uint32_t));

    NSUInteger thread_group_size = g_pipeline_string.maxTotalThreadsPerThreadgroup;
    if (thread_group_size > 256) thread_group_size = 256;

    id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:g_pipeline_string];
    [encoder setBuffer:str_buf offset:0 atIndex:0];
    [encoder setBuffer:hash_table_buf offset:0 atIndex:2];
    [encoder setBuffer:key_idx_buf offset:0 atIndex:3];
    [encoder setBuffer:count_buf offset:0 atIndex:4];
    [encoder setBytes:&table_size length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&max_str_len length:sizeof(uint32_t) atIndex:7];

    for (int64_t batch_offset = 0; batch_offset < num_rows; batch_offset += GPU_BATCH_SIZE) {
      int64_t batch_rows = std::min(GPU_BATCH_SIZE, num_rows - batch_offset);
      uint32_t batch_elements = (uint32_t)batch_rows;

      [encoder setBuffer:hdr_buf offset:batch_offset * sizeof(StringHeader) atIndex:1];
      [encoder setBytes:&batch_elements length:sizeof(uint32_t) atIndex:5];

      MTLSize grid_size = MTLSizeMake(batch_elements, 1, 1);
      MTLSize group_size = MTLSizeMake(thread_group_size, 1, 1);
      [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];

      if (batch_offset + GPU_BATCH_SIZE < num_rows) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    }

    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    uint32_t *count_ptr = (uint32_t *)[count_buf contents];
    auto end = std::chrono::high_resolution_clock::now();

    result.distinct_count = *count_ptr;
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.used_gpu = true;
    return 0;
  }
}

int gpu_count_distinct(const void *data, int64_t num_rows, GpuDataType data_type,
                       const GpuStringData *str_data, GpuCountDistinctResult &result)
{
  if (!init_metal()) {
    result.used_gpu = false;
    return -1;
  }
  if (num_rows < GPU_COUNT_DISTINCT_MIN_ROWS) {
    result.used_gpu = false;
    return -1;
  }
  switch (data_type) {
    case GpuDataType::INT64:
      return gpu_count_distinct_int64(static_cast<const int64_t *>(data), num_rows, result);
    case GpuDataType::DOUBLE:
      return gpu_count_distinct_double(static_cast<const double *>(data), num_rows, result);
    case GpuDataType::STRING:
      if (!str_data) return -1;
      return gpu_count_distinct_string(*str_data, num_rows, result);
    default:
      result.used_gpu = false;
      return -1;
  }
}

const char *get_gpu_count_distinct_init_error()
{
  return g_metal_init_error;
}

// Streaming: persistent hash table, insert batches then finish.
struct GpuCountDistinctStream {
  GpuDataType dtype = GpuDataType::INT64;
  uint32_t table_size = 0;
  id<MTLBuffer> hash_table_buf = nil;
  id<MTLBuffer> key_table_buf = nil;   // int64 for int64/double
  id<MTLBuffer> key_idx_buf = nil;     // uint32_t for string
  id<MTLBuffer> count_buf = nil;
  uint32_t max_str_len = 0;
  std::chrono::high_resolution_clock::time_point stream_start;
};

static int stream_run_int64_batch(GpuCountDistinctStream *s, const int64_t *data, int64_t n)
{
  if (!s || !s->hash_table_buf || n <= 0) return -1;
  id<MTLBuffer> input_buf = [g_device newBufferWithBytes:data
      length:(NSUInteger)(n * sizeof(int64_t)) options:MTLResourceStorageModeShared];
  if (!input_buf) return -1;
  uint32_t batch_elements = (uint32_t)n;
  NSUInteger thread_group_size = g_pipeline_int64.maxTotalThreadsPerThreadgroup;
  if (thread_group_size > 256) thread_group_size = 256;
  id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
  [encoder setComputePipelineState:g_pipeline_int64];
  [encoder setBuffer:input_buf offset:0 atIndex:0];
  [encoder setBuffer:s->hash_table_buf offset:0 atIndex:1];
  [encoder setBuffer:s->key_table_buf offset:0 atIndex:2];
  [encoder setBuffer:s->count_buf offset:0 atIndex:3];
  [encoder setBytes:&batch_elements length:sizeof(uint32_t) atIndex:4];
  [encoder setBytes:&s->table_size length:sizeof(uint32_t) atIndex:5];
  MTLSize grid_size = MTLSizeMake(batch_elements, 1, 1);
  MTLSize group_size = MTLSizeMake(thread_group_size, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
  [encoder endEncoding];
  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];
  return 0;
}

static int stream_run_double_batch(GpuCountDistinctStream *s, const double *data, int64_t n)
{
  if (!s || !s->hash_table_buf || n <= 0) return -1;
  const int64_t *data_as_i64 = reinterpret_cast<const int64_t *>(data);
  id<MTLBuffer> input_buf = [g_device newBufferWithBytes:data_as_i64
      length:(NSUInteger)(n * sizeof(int64_t)) options:MTLResourceStorageModeShared];
  if (!input_buf) return -1;
  uint32_t batch_elements = (uint32_t)n;
  NSUInteger thread_group_size = g_pipeline_double.maxTotalThreadsPerThreadgroup;
  if (thread_group_size > 256) thread_group_size = 256;
  id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
  [encoder setComputePipelineState:g_pipeline_double];
  [encoder setBuffer:input_buf offset:0 atIndex:0];
  [encoder setBuffer:s->hash_table_buf offset:0 atIndex:1];
  [encoder setBuffer:s->key_table_buf offset:0 atIndex:2];
  [encoder setBuffer:s->count_buf offset:0 atIndex:3];
  [encoder setBytes:&batch_elements length:sizeof(uint32_t) atIndex:4];
  [encoder setBytes:&s->table_size length:sizeof(uint32_t) atIndex:5];
  MTLSize grid_size = MTLSizeMake(batch_elements, 1, 1);
  MTLSize group_size = MTLSizeMake(thread_group_size, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
  [encoder endEncoding];
  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];
  return 0;
}

struct StringHeader { uint32_t offset; uint32_t length; };
static int stream_run_string_batch(GpuCountDistinctStream *s, const GpuStringData &str_data, int64_t n)
{
  if (!s || !s->hash_table_buf || n <= 0) return -1;
  uint64_t total_str_bytes = 0;
  if (n > 0) {
    total_str_bytes = str_data.offsets[n - 1] + str_data.lengths[n - 1];
  }
  const void *str_src = str_data.flat_buffer;
  static const char dummy_byte = 0;
  if (total_str_bytes == 0) {
    total_str_bytes = 1;
    str_src = &dummy_byte;
  }
  StringHeader *all_headers = new (std::nothrow) StringHeader[n];
  if (!all_headers) return -1;
  for (int64_t i = 0; i < n; i++) {
    all_headers[i].offset = str_data.offsets[i];
    all_headers[i].length = str_data.lengths[i];
  }
  id<MTLBuffer> str_buf = [g_device newBufferWithBytes:str_src
      length:(NSUInteger)total_str_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> hdr_buf = [g_device newBufferWithBytes:all_headers
      length:(NSUInteger)(n * sizeof(StringHeader)) options:MTLResourceStorageModeShared];
  delete[] all_headers;
  if (!str_buf || !hdr_buf) return -1;
  uint32_t batch_elements = (uint32_t)n;
  uint32_t max_len = str_data.max_str_len ? str_data.max_str_len : 1;
  NSUInteger thread_group_size = g_pipeline_string.maxTotalThreadsPerThreadgroup;
  if (thread_group_size > 256) thread_group_size = 256;
  id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
  [encoder setComputePipelineState:g_pipeline_string];
  [encoder setBuffer:str_buf offset:0 atIndex:0];
  [encoder setBuffer:hdr_buf offset:0 atIndex:1];
  [encoder setBuffer:s->hash_table_buf offset:0 atIndex:2];
  [encoder setBuffer:s->key_idx_buf offset:0 atIndex:3];
  [encoder setBuffer:s->count_buf offset:0 atIndex:4];
  [encoder setBytes:&batch_elements length:sizeof(uint32_t) atIndex:5];
  [encoder setBytes:&s->table_size length:sizeof(uint32_t) atIndex:6];
  [encoder setBytes:&max_len length:sizeof(uint32_t) atIndex:7];
  MTLSize grid_size = MTLSizeMake(batch_elements, 1, 1);
  MTLSize group_size = MTLSizeMake(thread_group_size, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
  [encoder endEncoding];
  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];
  return 0;
}

GpuCountDistinctStream *gpu_count_distinct_stream_create(GpuDataType dtype)
{
  if (!init_metal()) return nullptr;
  GpuCountDistinctStream *s = new (std::nothrow) GpuCountDistinctStream();
  if (!s) return nullptr;
  s->dtype = dtype;
  s->table_size = calc_table_size(GPU_MAX_TABLE_ENTRIES);
  s->hash_table_buf = [g_device newBufferWithLength:s->table_size * sizeof(uint32_t)
      options:MTLResourceStorageModeShared];
  if (!s->hash_table_buf) { delete s; return nullptr; }
  memset([s->hash_table_buf contents], 0, s->table_size * sizeof(uint32_t));
  if (dtype == GpuDataType::STRING) {
    s->key_idx_buf = [g_device newBufferWithLength:s->table_size * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    if (!s->key_idx_buf) { s->hash_table_buf = nil; delete s; return nullptr; }
    memset([s->key_idx_buf contents], 0, s->table_size * sizeof(uint32_t));
  } else {
    s->key_table_buf = [g_device newBufferWithLength:s->table_size * sizeof(int64_t)
        options:MTLResourceStorageModeShared];
    if (!s->key_table_buf) { s->hash_table_buf = nil; delete s; return nullptr; }
  }
  s->count_buf = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  if (!s->count_buf) {
    s->key_table_buf = nil;
    s->key_idx_buf = nil;
    s->hash_table_buf = nil;
    delete s;
    return nullptr;
  }
  memset([s->count_buf contents], 0, sizeof(uint32_t));
  s->stream_start = std::chrono::high_resolution_clock::now();
  return s;
}

int gpu_count_distinct_stream_insert_int64(GpuCountDistinctStream *s, const int64_t *data, int64_t n)
{
  if (!s || s->dtype != GpuDataType::INT64) return -1;
  for (int64_t offset = 0; offset < n; offset += GPU_BATCH_SIZE) {
    int64_t batch = std::min(GPU_BATCH_SIZE, n - offset);
    if (stream_run_int64_batch(s, data + offset, batch) != 0) return -1;
  }
  return 0;
}

int gpu_count_distinct_stream_insert_double(GpuCountDistinctStream *s, const double *data, int64_t n)
{
  if (!s || s->dtype != GpuDataType::DOUBLE) return -1;
  for (int64_t offset = 0; offset < n; offset += GPU_BATCH_SIZE) {
    int64_t batch = std::min(GPU_BATCH_SIZE, n - offset);
    if (stream_run_double_batch(s, data + offset, batch) != 0) return -1;
  }
  return 0;
}

int gpu_count_distinct_stream_insert_string(GpuCountDistinctStream *s, const GpuStringData &str_data,
                                            int64_t n)
{
  if (!s || s->dtype != GpuDataType::STRING) return -1;
  if (str_data.max_str_len > s->max_str_len) s->max_str_len = str_data.max_str_len;
  for (int64_t offset = 0; offset < n; offset += GPU_BATCH_SIZE) {
    int64_t batch = std::min(GPU_BATCH_SIZE, n - offset);
    GpuStringData batch_data;
    batch_data.flat_buffer = str_data.flat_buffer;
    batch_data.offsets = str_data.offsets + offset;
    batch_data.lengths = str_data.lengths + offset;
    batch_data.max_str_len = str_data.max_str_len;
    if (stream_run_string_batch(s, batch_data, batch) != 0) return -1;
  }
  return 0;
}

int gpu_count_distinct_stream_finish(GpuCountDistinctStream *s, GpuCountDistinctResult &result)
{
  if (!s) return -1;
  auto end = std::chrono::high_resolution_clock::now();
  result.distinct_count = *(uint32_t *)[s->count_buf contents];
  result.elapsed_ms = std::chrono::duration<double, std::milli>(end - s->stream_start).count();
  result.used_gpu = true;
  return 0;
}

void gpu_count_distinct_stream_destroy(GpuCountDistinctStream *s)
{
  if (!s) return;
  s->hash_table_buf = nil;
  s->key_table_buf = nil;
  s->key_idx_buf = nil;
  s->count_buf = nil;
  delete s;
}

} // end namespace gpu_acc
} // end namespace oceanbase

#else // not __APPLE__

namespace oceanbase
{
namespace gpu_acc
{

int gpu_count_distinct_int64(const int64_t *, int64_t, GpuCountDistinctResult &result)
{
  result.used_gpu = false;
  return -1;
}

int gpu_count_distinct_double(const double *, int64_t, GpuCountDistinctResult &result)
{
  result.used_gpu = false;
  return -1;
}

int gpu_count_distinct_string(const GpuStringData &, int64_t, GpuCountDistinctResult &result)
{
  result.used_gpu = false;
  return -1;
}

int gpu_count_distinct(const void *, int64_t, GpuDataType, const GpuStringData *,
                       GpuCountDistinctResult &result)
{
  result.used_gpu = false;
  return -1;
}

const char *get_gpu_count_distinct_init_error()
{
  return "";
}

GpuCountDistinctStream *gpu_count_distinct_stream_create(GpuDataType) { return nullptr; }
int gpu_count_distinct_stream_insert_int64(GpuCountDistinctStream *, const int64_t *, int64_t) { return -1; }
int gpu_count_distinct_stream_insert_double(GpuCountDistinctStream *, const double *, int64_t) { return -1; }
int gpu_count_distinct_stream_insert_string(GpuCountDistinctStream *, const GpuStringData &, int64_t) { return -1; }
int gpu_count_distinct_stream_finish(GpuCountDistinctStream *, GpuCountDistinctResult &) { return -1; }
void gpu_count_distinct_stream_destroy(GpuCountDistinctStream *) {}

} // end namespace gpu_acc
} // end namespace oceanbase

#endif // __APPLE__
