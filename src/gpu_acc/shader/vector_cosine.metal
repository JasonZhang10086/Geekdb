/*
 * Cosine distance kernels for vector GPU acceleration (Metal).
 * Used by ob_metal_cosine.mm.
 * cosine_distance = 1 - dot(a,b) / (|a|*|b|)
 */
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
