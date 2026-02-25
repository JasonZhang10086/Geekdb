/*
 * L2 squared distance kernels for vector GPU acceleration (Metal).
 * Used by ob_metal_l2.mm.
 */
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
