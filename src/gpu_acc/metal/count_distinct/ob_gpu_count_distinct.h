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

#ifndef OCEANBASE_GPU_ACC_OB_GPU_COUNT_DISTINCT_H_
#define OCEANBASE_GPU_ACC_OB_GPU_COUNT_DISTINCT_H_

#include <cstdint>
#include <cstddef>

namespace oceanbase
{
namespace gpu_acc
{

enum class GpuDataType
{
  INT64 = 0,
  DOUBLE = 1,
  STRING = 2,
};

struct GpuCountDistinctResult
{
  uint64_t distinct_count;
  double elapsed_ms;
  bool used_gpu;
};

struct GpuStringData
{
  const char *flat_buffer;
  const uint32_t *offsets;
  const uint32_t *lengths;
  uint32_t max_str_len;
};

int gpu_count_distinct_int64(const int64_t *data, int64_t num_rows,
                             GpuCountDistinctResult &result);

int gpu_count_distinct_double(const double *data, int64_t num_rows,
                              GpuCountDistinctResult &result);

int gpu_count_distinct_string(const GpuStringData &str_data, int64_t num_rows,
                              GpuCountDistinctResult &result);

int gpu_count_distinct(const void *data, int64_t num_rows, GpuDataType data_type,
                       const GpuStringData *str_data, GpuCountDistinctResult &result);

/** Returns a short reason string when init failed (for logging); empty if never failed or ready. */
const char *get_gpu_count_distinct_init_error();

/** Streaming API: persistent GPU hash table, insert batches then finish to get total distinct count. */
struct GpuCountDistinctStream;
GpuCountDistinctStream *gpu_count_distinct_stream_create(GpuDataType dtype);
int gpu_count_distinct_stream_insert_int64(GpuCountDistinctStream *s, const int64_t *data, int64_t n);
int gpu_count_distinct_stream_insert_double(GpuCountDistinctStream *s, const double *data, int64_t n);
int gpu_count_distinct_stream_insert_string(GpuCountDistinctStream *s, const GpuStringData &str_data,
                                            int64_t n);
int gpu_count_distinct_stream_finish(GpuCountDistinctStream *s, GpuCountDistinctResult &result);
void gpu_count_distinct_stream_destroy(GpuCountDistinctStream *s);

} // end namespace gpu_acc
} // end namespace oceanbase

#endif // OCEANBASE_GPU_ACC_OB_GPU_COUNT_DISTINCT_H_
