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

#ifndef OCEANBASE_GPU_ACC_VECTOR_METAL_OB_METAL_COSINE_H_
#define OCEANBASE_GPU_ACC_VECTOR_METAL_OB_METAL_COSINE_H_

#include <cstdint>

namespace oceanbase
{
namespace gpu_acc
{
namespace vector_metal
{

/**
 * Compute cosine distances between one query vector and N vectors on Metal (Mac GPU).
 * cosine_distance = 1 - dot(a,b) / (|a|*|b|)
 * distances_out[i] = 1 - dot(query, vectors[i]) / (norm(query) * norm(vectors[i]))
 */
int metal_cosine_query_vids(const float *query,
                            const float *vectors,
                            int64_t count,
                            int64_t dim,
                            float *distances_out);

/**
 * Cosine distance with offset-based layout: no CPU copy of vector data.
 * Row i is at vectors_data + byte_offsets[i].
 */
int metal_cosine_query_vids_strided(const float *query,
                                    const char *vectors_data,
                                    size_t vectors_data_byte_len,
                                    const uint32_t *byte_offsets,
                                    int64_t count,
                                    int64_t dim,
                                    float *distances_out);

/** Cosine strided with double elements (query and vectors_data as double*). */
int metal_cosine_query_vids_strided_f64(const double *query,
                                         const char *vectors_data,
                                         size_t vectors_data_byte_len,
                                         const uint32_t *byte_offsets,
                                         int64_t count,
                                         int64_t dim,
                                         float *distances_out);

/** Returns true if Metal is initialized and metal_cosine can run. */
bool is_metal_cosine_ready();

/** Returns a short error string when init failed; empty if ready or never failed. */
const char *get_metal_cosine_init_error();

/** Minimum count to consider using GPU (avoid overhead for tiny batches). */
static constexpr int64_t METAL_COSINE_MIN_COUNT = 256;

}  // namespace vector_metal
}  // namespace gpu_acc
}  // namespace oceanbase

#endif  // OCEANBASE_GPU_ACC_VECTOR_METAL_OB_METAL_COSINE_H_
