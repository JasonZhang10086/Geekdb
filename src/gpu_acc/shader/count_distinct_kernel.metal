/*
 * Count distinct kernels: lock-free GPU hash table (MurmurHash3 / FNV-1a).
 * Used by ob_gpu_count_distinct.mm.
 */
#include <metal_stdlib>
using namespace metal;

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
