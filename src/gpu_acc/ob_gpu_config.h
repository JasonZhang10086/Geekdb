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

#ifndef OCEANBASE_GPU_ACC_OB_GPU_CONFIG_H_
#define OCEANBASE_GPU_ACC_OB_GPU_CONFIG_H_

#include <cstddef>
#include <cstdint>

namespace oceanbase
{
namespace gpu_acc
{

static constexpr int64_t GPU_COUNT_DISTINCT_MIN_ROWS = 10000;

static constexpr int64_t GPU_HASH_TABLE_LOAD_FACTOR_PCT = 60;

/** Flush to GPU hash table when numeric (int64/double) buffer reaches this many bytes. */
static constexpr int64_t GPU_COUNT_DISTINCT_FLUSH_BYTES = 2 * 1024 * 1024;

/** Flush when row count reaches this (used for string row-count threshold). */
static constexpr int64_t GPU_COUNT_DISTINCT_FLUSH_ROWS = 256 * 1024;

/** Flush string buffer when flat buffer size reaches this many bytes. */
static constexpr size_t GPU_COUNT_DISTINCT_STR_FLUSH_BYTES = 4 * 1024 * 1024;

/** Minimum vector count to use Metal for IVF center assignment (vec_ivf_center_id). */
static constexpr int64_t VEC_IVF_METAL_MIN_COUNT = 32;

} // end namespace gpu_acc
} // end namespace oceanbase

#endif // OCEANBASE_GPU_ACC_OB_GPU_CONFIG_H_
