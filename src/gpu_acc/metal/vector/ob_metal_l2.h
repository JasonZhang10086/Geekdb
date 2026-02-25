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

#ifndef OCEANBASE_GPU_ACC_VECTOR_METAL_OB_METAL_L2_H_
#define OCEANBASE_GPU_ACC_VECTOR_METAL_OB_METAL_L2_H_

#include <cstdint>

namespace oceanbase
{
namespace gpu_acc
{
namespace vector_metal
{

/**
 * Compute L2 squared distances between one query vector and N vectors on Metal (Mac GPU).
 * Used by vector L2 distance expression (not vsag/HNSW).
 * distances_out[i] = sum_j (query[j] - vectors[i*dim + j])^2
 */
int metal_l2_query_vids(const float *query,
                        const float *vectors,
                        int64_t count,
                        int64_t dim,
                        float *distances_out);

/**
 * L2 squared with offset-based layout: no CPU copy of vector data.
 * Row i is at vectors_data + byte_offsets[i]. vectors_data_byte_len is the total byte length of vectors_data.
 */
int metal_l2_query_vids_strided(const float *query,
                                const char *vectors_data,
                                size_t vectors_data_byte_len,
                                const uint32_t *byte_offsets,
                                int64_t count,
                                int64_t dim,
                                float *distances_out);

/** L2 strided with double elements (query and vectors_data as double*). */
int metal_l2_query_vids_strided_f64(const double *query,
                                    const char *vectors_data,
                                    size_t vectors_data_byte_len,
                                    const uint32_t *byte_offsets,
                                    int64_t count,
                                    int64_t dim,
                                    float *distances_out);

/** L2 strided with int32 elements; kernel converts to float. query and row data as int32_t*. */
int metal_l2_query_vids_strided_i32(const int32_t *query,
                                    const char *vectors_data,
                                    size_t vectors_data_byte_len,
                                    const uint32_t *byte_offsets,
                                    int64_t count,
                                    int64_t dim,
                                    float *distances_out);

/** Returns true on Apple if Metal is initialized and metal_l2 can run; non-Apple always false. */
bool is_metal_ready();

/** Returns a short error string when init failed; empty if ready or never failed. */
const char *get_metal_l2_init_error();

/** Minimum count to consider using GPU (avoid overhead for tiny batches). */
static constexpr int64_t METAL_L2_MIN_COUNT = 256;

/**
 * Batch L2 squared: M queries vs N vectors, output M*N distances.
 * queries: M * dim (contiguous), vectors: N * dim (contiguous).
 * distances_out[i*num_vectors + j] = L2^2(queries[i], vectors[j]).
 * Used by IVF kmeans assign (samples vs centers) for one-shot GPU matrix.
 */
int metal_l2_batch_query_vids(const float *queries,
                              const float *vectors,
                              int64_t num_queries,
                              int64_t num_vectors,
                              int64_t dim,
                              float *distances_out);

}  // namespace vector_metal
}  // namespace gpu_acc
}  // namespace oceanbase

#endif  // OCEANBASE_GPU_ACC_VECTOR_METAL_OB_METAL_L2_H_
