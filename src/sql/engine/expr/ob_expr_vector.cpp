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

#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/expr/ob_expr_vector.h"
#include "sql/engine/expr/ob_array_expr_utils.h"
#include "sql/engine/ob_exec_context.h"
#include "share/vector/ob_i_vector.h"
#include "share/vector/ob_continuous_format.h"
#include "share/vector/ob_fixed_length_base.h"
#include "share/vector_type/ob_vector_norm.h"
#ifdef __APPLE__
#include "gpu_acc/metal/vector/ob_metal_l2.h"
#include "gpu_acc/metal/vector/ob_metal_cosine.h"
#include <stdlib.h>  // posix_memalign for Metal unified-memory output
#endif

namespace oceanbase
{
namespace sql
{

ObExprVectorDistance::SparseVectorDisFunc::FuncPtrType ObExprVectorDistance::SparseVectorDisFunc::spiv_distance_funcs[] = 
{
  nullptr, // cosine_distance
  ObSparseVectorIpDistance::spiv_ip_distance_func,
  nullptr, // l2_distance
  nullptr, // l1_distance
  nullptr, // l2_square
  nullptr,
};

ObExprVector::ObExprVector(ObIAllocator &alloc,
                         ObExprOperatorType type,
                         const char *name,
                         int32_t param_num, 
                         int32_t dimension) : ObFuncExprOperator(alloc, type, name, param_num, VALID_FOR_GENERATED_COL, dimension) 
{
}
// [a,b,c,...] is array type, there is no dim_cnt_ in ObCollectionArrayType
int ObExprVector::calc_result_type2(
    ObExprResType &type,
    ObExprResType &type1,
    ObExprResType &type2,
    common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;
  uint16_t unused_id = UINT16_MAX;
  if (OB_FAIL(ObArrayExprUtils::calc_cast_type2(type_, type1, type2, type_ctx, unused_id))) {
    LOG_WARN("failed to calc cast type", K(ret), K(type1));
  } else {
    type.set_type(ObDoubleType);
    type.set_calc_type(ObDoubleType);
  }
  return ret;
}

int ObExprVector::calc_result_type1(
    ObExprResType &type,
    ObExprResType &type1,
    common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS; 
  if (OB_FAIL(ObArrayExprUtils::calc_cast_type(type_, type1, type_ctx))) {
    LOG_WARN("failed to calc cast type", K(ret), K(type1));
  } else {
    type.set_type(ObDoubleType);
    type.set_calc_type(ObDoubleType);
  }
  return ret;
}

ObExprVectorDistance::ObExprVectorDistance(ObIAllocator &alloc)
    : ObExprVector(alloc, T_FUN_SYS_VECTOR_DISTANCE, N_VECTOR_DISTANCE, TWO_OR_THREE, NOT_ROW_DIMENSION)
{}

ObExprVectorDistance::ObExprVectorDistance(
    ObIAllocator &alloc,
    ObExprOperatorType type,
    const char *name,
    int32_t param_num, 
    int32_t dimension)
      : ObExprVector(alloc, type, name, param_num, dimension)
{}

int ObExprVectorDistance::calc_result_typeN(
    ObExprResType &type,
    ObExprResType *types_stack,
    int64_t param_num,
    common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else if (OB_FAIL(calc_result_type2(type, types_stack[0], types_stack[1], type_ctx))) {
    LOG_WARN("failed to calc result type", K(ret));
  }
  return ret;
}

int ObExprVectorDistance::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                  ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  rt_expr.eval_func_ = ObExprVectorDistance::calc_distance;
  return ret;
}

int ObExprVectorDistance::calc_distance(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  int ret = OB_SUCCESS;
  ObVecDisType dis_type = ObVecDisType::EUCLIDEAN; // default metric
  if (3 == expr.arg_cnt_) {
    ObDatum *datum = NULL;
    if (OB_FAIL(expr.args_[2]->eval(ctx, datum))) {
      LOG_WARN("eval failed", K(ret));
    } else if (datum->is_null()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_WARN("invalid arg", K(ret), K(*datum));
    } else {
      dis_type = static_cast<ObVecDisType>(datum->get_int());
    }
  }
  if (FAILEDx(calc_distance(expr, ctx, res_datum, dis_type))) {
    LOG_WARN("failed to calc distance", K(ret), K(dis_type));
  }
  return ret;
}

int ObExprVectorDistance::calc_distance(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum, ObVecDisType dis_type)
{
  int ret = OB_SUCCESS;
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &tmp_allocator = tmp_alloc_g.get_allocator();
  ObIArrayType *arr_l = NULL;
  ObIArrayType *arr_r = NULL;
  bool contain_null = false;
  double distance = 0.0;
  if (dis_type < ObVecDisType::COSINE || dis_type >= ObVecDisType::MAX_TYPE) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpect distance type", K(ret), K(dis_type));
  } else if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[0]), ctx, tmp_allocator, arr_l, contain_null))) {
    LOG_WARN("failed to get vector", K(ret), K(*expr.args_[0]));
  } else if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[1]), ctx, tmp_allocator, arr_r, contain_null))) {
    LOG_WARN("failed to get vector", K(ret), K(*expr.args_[1]));
  } else if (contain_null) {
    res_datum.set_null();
  } else if (OB_ISNULL(arr_l) || OB_ISNULL(arr_r)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected nullptr", K(ret), K(arr_l), K(arr_r));
  } else if ((arr_l->get_array_type()->is_sparse_vector_type() && !arr_r->get_array_type()->is_sparse_vector_type()) 
             || (!arr_l->get_array_type()->is_sparse_vector_type() && arr_r->get_array_type()->is_sparse_vector_type())) {
    LOG_WARN("calc distance for sparse vector and other type is not supported", K(ret));
  } else if (arr_l->get_array_type()->is_sparse_vector_type() && arr_r->get_array_type()->is_sparse_vector_type()) {
    const ObMapType *spv_l = dynamic_cast<const ObMapType *>(arr_l);
    const ObMapType *spv_r = dynamic_cast<const ObMapType *>(arr_r);
    if (OB_ISNULL(spv_l) || OB_ISNULL(spv_r)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("sparse vector type cast failed", K(ret));
    } else if (dis_type != ObVecDisType::DOT) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("sparse vector not support", K(ret), K(dis_type));
    } else if (OB_FAIL(SparseVectorDisFunc::spiv_distance_funcs[dis_type](spv_l, spv_r, distance))) {
      LOG_WARN("sparse vector failed to calc distance", K(ret), K(dis_type));
    } else {
      if (dis_type == ObVecDisType::EUCLIDEAN || dis_type == ObVecDisType::EUCLIDEAN_SQUARED || dis_type == ObVecDisType::COSINE) {
        res_datum.set_float(static_cast<float>(distance));
      } else {
        res_datum.set_double(distance);
      }
    }
  } else {
    if (OB_UNLIKELY(arr_l->size() != arr_r->size())) {
      ret = OB_ERR_INVALID_VECTOR_DIM;
      LOG_WARN("check array validty failed", K(ret), K(arr_l->size()), K(arr_r->size()));
    } else if (arr_l->contain_null() || arr_r->contain_null()) {
      ret = OB_ERR_NULL_VALUE;
      LOG_WARN("array with null can't calculate vector distance", K(ret));
    } else {
      const float *data_l = reinterpret_cast<const float*>(arr_l->get_data());
      const float *data_r = reinterpret_cast<const float*>(arr_r->get_data());
      const uint32_t size = arr_l->size();
      if (DisFunc<float>::distance_funcs[dis_type] == nullptr) {
        ret = OB_NOT_SUPPORTED;
        LOG_WARN("not support", K(ret), K(dis_type));
      } else if (OB_FAIL(DisFunc<float>::distance_funcs[dis_type](data_l, data_r, size, distance))) {
        if (OB_ERR_NULL_VALUE == ret) {
          res_datum.set_null();
          ret = OB_SUCCESS; // ignore
        } else {
          LOG_WARN("failed to calc distance", K(ret), K(dis_type));
        }
      } else {
        if (dis_type == ObVecDisType::EUCLIDEAN || dis_type == ObVecDisType::EUCLIDEAN_SQUARED || dis_type == ObVecDisType::COSINE) {
          res_datum.set_float(static_cast<float>(distance));
        } else {
          res_datum.set_double(distance);
        }
      }
    }
  }
  
  return ret;
}

ObExprVectorL1Distance::ObExprVectorL1Distance(ObIAllocator &alloc)
    : ObExprVectorDistance(alloc, T_FUN_SYS_L1_DISTANCE, N_VECTOR_L1_DISTANCE, 2, NOT_ROW_DIMENSION) {}

int ObExprVectorL1Distance::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                    ObExpr &rt_expr) const
{
    int ret = OB_SUCCESS;  
    rt_expr.eval_func_ = ObExprVectorL1Distance::calc_l1_distance;
    return ret;
}

int ObExprVectorL1Distance::calc_l1_distance(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  return ObExprVectorDistance::calc_distance(expr, ctx, res_datum, ObVecDisType::MANHATTAN);
}

// ObExprL2MetalExtraInfo: CG 阶段判定 Metal 是否可用，避免运行时每轮判断
int ObExprL2MetalExtraInfo::serialize(char *buf, const int64_t len, int64_t &pos) const
{
  int ret = OB_SUCCESS;
  const int64_t buf_len = len;
  OB_UNIS_ENCODE(use_metal_);
  return ret;
}

int ObExprL2MetalExtraInfo::deserialize(const char *buf, const int64_t len, int64_t &pos)
{
  int ret = OB_SUCCESS;
  const int64_t data_len = len;
  OB_UNIS_DECODE(use_metal_);
  return ret;
}

int64_t ObExprL2MetalExtraInfo::get_serialize_size() const
{
  int64_t len = 0;
  OB_UNIS_ADD_LEN(use_metal_);
  return len;
}

int ObExprL2MetalExtraInfo::deep_copy(common::ObIAllocator &allocator,
                                      const ObExprOperatorType type,
                                      ObIExprExtraInfo *&copied_info) const
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObExprExtraInfoFactory::alloc(allocator, type, copied_info))) {
    LOG_WARN("alloc ObExprL2MetalExtraInfo failed", K(ret));
  } else {
    static_cast<ObExprL2MetalExtraInfo *>(copied_info)->use_metal_ = use_metal_;
  }
  return ret;
}

int ObExprCosineMetalExtraInfo::serialize(char *buf, const int64_t len, int64_t &pos) const
{
  int ret = OB_SUCCESS;
  const int64_t buf_len = len;
  OB_UNIS_ENCODE(use_metal_);
  return ret;
}

int ObExprCosineMetalExtraInfo::deserialize(const char *buf, const int64_t len, int64_t &pos)
{
  int ret = OB_SUCCESS;
  const int64_t data_len = len;
  OB_UNIS_DECODE(use_metal_);
  return ret;
}

int64_t ObExprCosineMetalExtraInfo::get_serialize_size() const
{
  int64_t len = 0;
  OB_UNIS_ADD_LEN(use_metal_);
  return len;
}

int ObExprCosineMetalExtraInfo::deep_copy(common::ObIAllocator &allocator,
                                         const ObExprOperatorType type,
                                         ObIExprExtraInfo *&copied_info) const
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObExprExtraInfoFactory::alloc(allocator, type, copied_info))) {
    LOG_WARN("alloc ObExprCosineMetalExtraInfo failed", K(ret));
  } else {
    static_cast<ObExprCosineMetalExtraInfo *>(copied_info)->use_metal_ = use_metal_;
  }
  return ret;
}

ObExprVectorL2Distance::ObExprVectorL2Distance(ObIAllocator &alloc)
    : ObExprVectorDistance(alloc, T_FUN_SYS_L2_DISTANCE, N_VECTOR_L2_DISTANCE, 2, NOT_ROW_DIMENSION) {}

int ObExprVectorL2Distance::calc_result_type2(
    ObExprResType &type,
    ObExprResType &type1,
    ObExprResType &type2,
    common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;
  uint16_t unused_id = UINT16_MAX;
  if (OB_FAIL(ObArrayExprUtils::calc_cast_type2(type_, type1, type2, type_ctx, unused_id))) {
    LOG_WARN("failed to calc cast type", K(ret), K(type1));
  } else {
    type.set_type(ObFloatType);
    type.set_calc_type(ObFloatType);
  }
  return ret;
}

int ObExprVectorL2Distance::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                    ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  rt_expr.eval_func_ = ObExprVectorL2Distance::calc_l2_distance;
  ObExprL2MetalExtraInfo *extra = nullptr;
  if (OB_FAIL(ObExprExtraInfoFactory::alloc(*expr_cg_ctx.allocator_, T_FUN_SYS_L2_DISTANCE, reinterpret_cast<ObIExprExtraInfo *&>(extra)))) {
    LOG_WARN("alloc L2 metal extra info failed", K(ret));
  } else {
#ifdef __APPLE__
    extra->use_metal_ = gpu_acc::vector_metal::is_metal_ready();
#else
    extra->use_metal_ = false;
#endif
    rt_expr.extra_info_ = extra;
    if (extra->use_metal_) {
      rt_expr.eval_batch_func_ = ObExprVectorL2Distance::calc_l2_distance_batch_metal;
      rt_expr.eval_vector_func_ = ObExprVectorL2Distance::calc_l2_distance_vector_metal;
    } else {
      rt_expr.eval_batch_func_ = ObExprVectorL2Distance::calc_l2_distance_batch_cpu;
      rt_expr.eval_vector_func_ = ObExprVectorL2Distance::calc_l2_distance_vector_cpu;
    }
  }
  return ret;
}

static int64_t g_l2_single_call_cnt = 0;
int ObExprVectorL2Distance::calc_l2_distance(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  int ret = OB_SUCCESS;
  g_l2_single_call_cnt++;
  if (REACH_TIME_INTERVAL(2000000)) {
    LOG_ERROR("GPU_TRACE: calc_l2_distance single-row called", K(g_l2_single_call_cnt));
  }
  ret = ObExprVectorDistance::calc_distance(expr, ctx, res_datum, ObExprVectorDistance::ObVecDisType::EUCLIDEAN);
  return ret;
}

static int l2_batch_cpu_loop(const ObExpr &expr, ObEvalCtx &ctx, const ObBitVector &skip,
                             const int64_t batch_size, ObDatum *results, ObBitVector &eval_flags)
{
  int ret = OB_SUCCESS;
  for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; i++) {
    if (skip.at(i) || eval_flags.at(i)) continue;
    ObDatum &res = results[i];
    if (OB_FAIL(ObExprVectorDistance::calc_distance(expr, ctx, res, ObExprVectorDistance::ObVecDisType::EUCLIDEAN))) {
      LOG_WARN("calc_distance failed", K(ret), K(i));
    } else {
      eval_flags.set(i);
    }
  }
  return ret;
}

int ObExprVectorL2Distance::calc_l2_distance_batch_cpu(const ObExpr &expr, ObEvalCtx &ctx,
                                                       const ObBitVector &skip, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(expr.args_[0]->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval batch arg0 failed", K(ret));
  } else if (OB_FAIL(expr.args_[1]->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval batch arg1 failed", K(ret));
  } else {
    ObDatum *results = expr.locate_batch_datums(ctx);
    ObDatum *d0 = expr.args_[0]->locate_batch_datums(ctx);
    if (OB_ISNULL(results) || OB_ISNULL(d0)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("batch datum is null", K(ret));
    } else {
      ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
      ret = l2_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
    }
  }
  return ret;
}

#ifdef __APPLE__
int ObExprVectorL2Distance::calc_l2_distance_batch_metal(const ObExpr &expr, ObEvalCtx &ctx,
                                                         const ObBitVector &skip, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(expr.args_[0]->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval batch arg0 failed", K(ret));
  } else if (OB_FAIL(expr.args_[1]->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval batch arg1 failed", K(ret));
  } else {
    ObDatum *results = expr.locate_batch_datums(ctx);
    ObDatum *d0 = expr.args_[0]->locate_batch_datums(ctx);
    ObDatum *d1 = expr.args_[1]->locate_batch_datums(ctx);
    if (OB_ISNULL(results) || OB_ISNULL(d0) || OB_ISNULL(d1)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("batch datum is null", K(ret));
    } else {
      ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
      int64_t valid_cnt = 0;
      for (int64_t i = 0; i < batch_size; i++) {
        if (!skip.at(i) && !eval_flags.at(i)) valid_cnt++;
      }
      bool do_metal = (valid_cnt >= static_cast<int64_t>(gpu_acc::vector_metal::METAL_L2_MIN_COUNT));
      if (!do_metal || valid_cnt <= 0) {
        ret = l2_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
      } else {
        ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
        common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();
        ObIArrayType *query_arr = nullptr;
        bool contain_null = false;
        if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[1]), ctx, alloc, query_arr, contain_null))
            || OB_ISNULL(query_arr) || contain_null || query_arr->size() == 0) {
          ret = l2_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
        } else {
          const int64_t dim = static_cast<int64_t>(query_arr->size());
          const float *query_ptr = reinterpret_cast<const float *>(query_arr->get_data());
          const uint64_t vec_buf_size = static_cast<uint64_t>(valid_cnt) * static_cast<uint64_t>(dim) * sizeof(float);
          const uint64_t dist_buf_size = static_cast<uint64_t>(valid_cnt) * sizeof(float);
          float *vectors_buf = static_cast<float *>(alloc.alloc(vec_buf_size));
          void *batch_aligned_ptr = nullptr;
          float *distances_buf = nullptr;
          if (0 == posix_memalign(&batch_aligned_ptr, 4096, dist_buf_size)) {
            distances_buf = static_cast<float *>(batch_aligned_ptr);
          } else {
            distances_buf = static_cast<float *>(alloc.alloc(dist_buf_size));
          }
          if (OB_ISNULL(vectors_buf) || OB_ISNULL(distances_buf)) {
            ret = l2_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
          } else {
            int64_t compact = 0;
            for (int64_t i = 0; OB_SUCC(ret) && compact < valid_cnt && i < batch_size; i++) {
              if (skip.at(i) || eval_flags.at(i)) continue;
              if (d0[i].is_null()) {
                results[i].set_null();
                eval_flags.set(i);
                continue;
              }
              ObIArrayType *row_arr = nullptr;
              if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[0]), d0[i], ctx, alloc, row_arr))) {
                LOG_WARN("get_type_vector failed", K(ret), K(i));
                break;
              }
              if (OB_ISNULL(row_arr) || row_arr->size() != query_arr->size()) {
                ret = OB_ERR_UNEXPECTED;
                LOG_WARN("vector dim mismatch", K(ret), K(i));
                break;
              }
              memcpy(vectors_buf + compact * dim, row_arr->get_data(), static_cast<size_t>(dim) * sizeof(float));
              compact++;
            }
            if (OB_SUCC(ret) && compact >= static_cast<int64_t>(gpu_acc::vector_metal::METAL_L2_MIN_COUNT)) {
              int gpu_ret = gpu_acc::vector_metal::metal_l2_query_vids(
                  query_ptr, vectors_buf, compact, dim, distances_buf);
              if (gpu_ret == 0) {
                compact = 0;
                for (int64_t i = 0; i < batch_size; i++) {
                  if (skip.at(i) || eval_flags.at(i)) continue;
                  if (d0[i].is_null()) continue;
                  results[i].set_float(static_cast<float>(sqrt(static_cast<double>(distances_buf[compact]))));
                  eval_flags.set(i);
                  compact++;
                }
              } else {
                ret = l2_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
              }
            } else {
              ret = l2_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
            }
            if (batch_aligned_ptr != nullptr) {
              free(batch_aligned_ptr);
            }
          }
        }
      }
    }
  }
  return ret;
}
#else
int ObExprVectorL2Distance::calc_l2_distance_batch_metal(const ObExpr &expr, ObEvalCtx &ctx,
                                                         const ObBitVector &skip, const int64_t batch_size)
{
  return calc_l2_distance_batch_cpu(expr, ctx, skip, batch_size);
}
#endif

// 从 vector 取第 idx 行并解析为 ObIArrayType，不调用 eval()/datum，避免与 VEC_FIXED 等格式不同步导致崩溃
static int get_array_from_vector(const ObExpr &arg_expr, common::ObIVector *vec, int64_t idx,
                                 ObEvalCtx &ctx, common::ObIAllocator &alloc,
                                 ObIArrayType *&arr, bool &is_null)
{
  int ret = OB_SUCCESS;
  arr = nullptr;
  is_null = false;
  if (OB_ISNULL(vec)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("vector is null", K(ret));
  } else if (vec->is_null(idx)) {
    is_null = true;
  } else {
    ObString s = vec->get_string(idx);
    ObDatum d;
    d.set_string(s);
    if (OB_FAIL(ObArrayExprUtils::get_type_vector(arg_expr, d, ctx, alloc, arr))) {
      LOG_WARN("get_type_vector from datum failed", K(ret), K(idx));
    }
  }
  return ret;
}

// Compact 格式 vector：payload 前 4 字节为 version，之后为连续 float；直接返回指针避免 get_array_from_vector 的解析与拷贝
static bool get_compact_vector_payload(common::ObIVector *vec, int64_t idx,
                                       const char *&payload, int32_t &payload_len)
{
  if (OB_ISNULL(vec) || vec->is_null(idx)) return false;
  ObLength len = 0;
  vec->get_payload(idx, payload, len);
  payload_len = static_cast<int32_t>(len);
  if (payload_len < 4 || payload == nullptr) return false;
  if (!ObCollectionExprUtil::is_compact_fmt_cell(payload)) return false;
  if ((payload_len - 4) % sizeof(float) != 0) return false;
  payload += 4;
  payload_len -= 4;
  return true;
}

// 仅做逐行 CPU L2，供 vector_cpu 与 vector_metal 失败时回退
static int l2_vector_cpu_loop(const ObExpr &expr, ObEvalCtx &ctx, const ObBitVector &skip,
                               const EvalBound &bound, common::ObIVector *res_vec,
                               common::ObIVector *vec0, common::ObIVector *vec1,
                               common::ObArenaAllocator &alloc)
{
  int ret = OB_SUCCESS;
  ObIArrayType *query_arr = nullptr;
  bool query_null = false;
  if (OB_FAIL(get_array_from_vector(*(expr.args_[1]), vec1, bound.start(), ctx, alloc, query_arr, query_null))) {
    LOG_WARN("get query array from vector failed", K(ret));
  } else if (query_null || OB_ISNULL(query_arr) || query_arr->size() == 0) {
    ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
    for (int64_t i = bound.start(); OB_SUCC(ret) && i < bound.end(); i++) {
      if (skip.at(i) || eval_flags.at(i)) continue;
      res_vec->set_null(i);
      eval_flags.set(i);
    }
  } else {
    const float *query_data = reinterpret_cast<const float *>(query_arr->get_data());
    const uint32_t dim = query_arr->size();
    ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
    const bool query_contain_null = query_arr->contain_null();

    // 一个 vector 只有一种 type，所有格式（含 VEC_FIXED / VEC_UNIFORM / VEC_UNIFORM_CONST）只取一次 type，循环内不再 get_type
    common::VectorFormat fmt0 = vec0->get_format();
    const char *vec0_data_ptr = nullptr;
    int64_t vec0_row_bytes = 0;
    bool use_fixed_path = false;
    if (fmt0 == common::VEC_FIXED) {
      common::ObFixedLengthBase *fix_base = static_cast<common::ObFixedLengthBase *>(vec0);
      vec0_data_ptr = fix_base->get_data();
      vec0_row_bytes = static_cast<int64_t>(fix_base->get_length());
      if (vec0_data_ptr != nullptr && vec0_row_bytes > 0
          && static_cast<int64_t>(dim) * static_cast<int64_t>(sizeof(float)) == vec0_row_bytes) {
        use_fixed_path = true;
      }
    }

    // 完全不拿 type：VEC_FIXED 用直接指针，compact 用 get_compact_vector_payload，仅非 compact 时才按行 get_array_from_vector
    for (int64_t i = bound.start(); OB_SUCC(ret) && i < bound.end(); i++) {
      if (skip.at(i) || eval_flags.at(i)) continue;
      if (vec0->is_null(i)) {
        res_vec->set_null(i);
        eval_flags.set(i);
        continue;
      }

      const float *row_data = nullptr;
      uint32_t row_dim = 0;
      bool row_has_null = false;

      if (use_fixed_path) {
        row_data = reinterpret_cast<const float *>(vec0_data_ptr + i * vec0_row_bytes);
        row_dim = dim;
        row_has_null = false;
      } else {
        const char *row_payload = nullptr;
        int32_t row_len = 0;
        if (get_compact_vector_payload(vec0, i, row_payload, row_len)
            && row_len > 0 && (row_len / static_cast<int32_t>(sizeof(float)) == static_cast<int32_t>(dim))) {
          row_data = reinterpret_cast<const float *>(row_payload);
          row_dim = dim;
          row_has_null = false;
        } else {
          ObIArrayType *row_arr = nullptr;
          bool row_null = false;
          if (OB_FAIL(get_array_from_vector(*(expr.args_[0]), vec0, i, ctx, alloc, row_arr, row_null))) {
            LOG_WARN("get_array_from_vector failed", K(ret), K(i));
            continue;
          }
          if (row_null || OB_ISNULL(row_arr)) {
            res_vec->set_null(i);
            eval_flags.set(i);
            continue;
          }
          if (row_arr->size() != dim) {
            ret = OB_ERR_INVALID_VECTOR_DIM;
            LOG_WARN("vector dim mismatch", K(ret), K(row_arr->size()), K(dim));
            continue;
          }
          row_has_null = row_arr->contain_null();
          row_data = reinterpret_cast<const float *>(row_arr->get_data());
          row_dim = row_arr->size();
        }
      }

      if (row_has_null || query_contain_null) {
        res_vec->set_null(i);
        eval_flags.set(i);
      } else if (row_data != nullptr && row_dim == dim) {
        double distance = 0.0;
        if (ObExprVectorDistance::DisFunc<float>::distance_funcs[ObExprVectorDistance::ObVecDisType::EUCLIDEAN] == nullptr) {
          ret = OB_NOT_SUPPORTED;
          LOG_WARN("L2 distance func not supported", K(ret));
        } else if (OB_FAIL(ObExprVectorDistance::DisFunc<float>::distance_funcs[ObExprVectorDistance::ObVecDisType::EUCLIDEAN](row_data, query_data, dim, distance))) {
          if (ret == OB_ERR_NULL_VALUE) {
            res_vec->set_null(i);
            eval_flags.set(i);
            ret = OB_SUCCESS;
          } else {
            LOG_WARN("calc L2 distance failed", K(ret), K(i));
          }
        } else {
          res_vec->set_float(i, static_cast<float>(distance));
          eval_flags.set(i);
        }
      }
    }
  }
  return ret;
}

int ObExprVectorL2Distance::calc_l2_distance_vector_cpu(VECTOR_EVAL_FUNC_ARG_DECL)
{
  int ret = OB_SUCCESS;
  const int64_t batch_size = bound.batch_size();
  if (OB_FAIL(expr.args_[0]->eval_vector(ctx, skip, bound))) {
    LOG_WARN("eval_vector arg0 failed", K(ret));
  } else if (OB_FAIL(expr.args_[1]->eval_vector(ctx, skip, bound))) {
    LOG_WARN("eval_vector arg1 failed", K(ret));
  } else if (OB_FAIL(expr.init_vector_for_write(ctx, VEC_UNIFORM, batch_size))) {
    LOG_WARN("init_vector_for_write failed", K(ret));
  } else {
    common::ObIVector *res_vec = expr.get_vector(ctx);
    common::ObIVector *vec0 = expr.args_[0]->get_vector(ctx);
    common::ObIVector *vec1 = expr.args_[1]->get_vector(ctx);
    if (OB_ISNULL(res_vec) || OB_ISNULL(vec0) || OB_ISNULL(vec1)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("get_vector returned null", K(ret));
    } else {
      ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
      common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();
      ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
      batch_guard.set_batch_size(batch_size);
      ret = l2_vector_cpu_loop(expr, ctx, skip, bound, res_vec, vec0, vec1, alloc);
    }
  }
  return ret;
}

#ifdef __APPLE__
int ObExprVectorL2Distance::calc_l2_distance_vector_metal(VECTOR_EVAL_FUNC_ARG_DECL)
{
  int ret = OB_SUCCESS;
  const int64_t batch_size = bound.batch_size();
  if (OB_FAIL(expr.args_[0]->eval_vector(ctx, skip, bound))) {
    LOG_WARN("eval_vector arg0 failed", K(ret));
  } else if (OB_FAIL(expr.args_[1]->eval_vector(ctx, skip, bound))) {
    LOG_WARN("eval_vector arg1 failed", K(ret));
  } else if (OB_FAIL(expr.init_vector_for_write(ctx, VEC_UNIFORM, batch_size))) {
    LOG_WARN("init_vector_for_write failed", K(ret));
  } else {
    common::ObIVector *res_vec = expr.get_vector(ctx);
    common::ObIVector *vec0 = expr.args_[0]->get_vector(ctx);
    common::ObIVector *vec1 = expr.args_[1]->get_vector(ctx);
    if (OB_ISNULL(res_vec) || OB_ISNULL(vec0) || OB_ISNULL(vec1)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("get_vector returned null", K(ret));
    } else {
      ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
      ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
      common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();
      ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
      batch_guard.set_batch_size(batch_size);

      int64_t valid_cnt = 0;
      for (int64_t i = bound.start(); i < bound.end(); i++) {
        if (!skip.at(i) && !eval_flags.at(i)) valid_cnt++;
      }
      bool do_metal = (valid_cnt >= static_cast<int64_t>(gpu_acc::vector_metal::METAL_L2_MIN_COUNT));
      if (!do_metal || valid_cnt <= 0) {
        ret = l2_vector_cpu_loop(expr, ctx, skip, bound, res_vec, vec0, vec1, alloc);
      } else {
        bool metal_ok = true;
        const float *query_ptr = nullptr;
        const double *query_ptr_f64 = nullptr;
        int64_t dim = 0;
        int elem_size = 4;
        const char *query_payload = nullptr;
        int32_t query_payload_len = 0;
        bool query_compact = get_compact_vector_payload(vec1, bound.start(), query_payload, query_payload_len);
        if (query_compact && query_payload_len > 0) {
          common::VectorFormat fmt0 = vec0->get_format();
          if (fmt0 == common::VEC_FIXED) {
            common::ObFixedLengthBase *fix_base = static_cast<common::ObFixedLengthBase *>(vec0);
            const int64_t row_bytes = static_cast<int64_t>(fix_base->get_length());
            if (row_bytes > 0 && query_payload_len == row_bytes) {
              if (row_bytes % sizeof(float) == 0) {
                dim = row_bytes / static_cast<int64_t>(sizeof(float));
                query_ptr = reinterpret_cast<const float *>(query_payload);
                elem_size = 4;
              } else if (row_bytes % sizeof(double) == 0) {
                dim = row_bytes / static_cast<int64_t>(sizeof(double));
                query_ptr_f64 = reinterpret_cast<const double *>(query_payload);
                elem_size = 8;
              }
            }
          }
          if (dim <= 0) {
            dim = static_cast<int64_t>(query_payload_len) / static_cast<int64_t>(sizeof(float));
            if (dim > 0 && static_cast<int64_t>(query_payload_len) == dim * static_cast<int64_t>(sizeof(float))) {
              query_ptr = reinterpret_cast<const float *>(query_payload);
              elem_size = 4;
            } else {
              dim = static_cast<int64_t>(query_payload_len) / static_cast<int64_t>(sizeof(double));
              if (dim > 0 && static_cast<int64_t>(query_payload_len) == dim * static_cast<int64_t>(sizeof(double))) {
                query_ptr_f64 = reinterpret_cast<const double *>(query_payload);
                elem_size = 8;
              } else {
                query_ptr = reinterpret_cast<const float *>(query_payload);
                dim = static_cast<int64_t>(query_payload_len) / static_cast<int64_t>(sizeof(float));
                elem_size = 4;
              }
            }
          }
        }
        if (query_ptr == nullptr && query_ptr_f64 == nullptr) {
          ObIArrayType *query_arr = nullptr;
          bool query_null = false;
          if (OB_FAIL(get_array_from_vector(*(expr.args_[1]), vec1, bound.start(), ctx, alloc, query_arr, query_null))
              || query_null || OB_ISNULL(query_arr) || query_arr->size() == 0) {
            metal_ok = false;
          } else {
            dim = static_cast<int64_t>(query_arr->size());
            query_ptr = reinterpret_cast<const float *>(query_arr->get_data());
            elem_size = 4;
          }
        }
        if (metal_ok && dim > 0 && (query_ptr != nullptr || query_ptr_f64 != nullptr)) {
          const uint64_t dist_buf_size = static_cast<uint64_t>(valid_cnt) * sizeof(float);
          float *distances_buf = nullptr;
          void *aligned_ptr = nullptr;
          if (0 == posix_memalign(&aligned_ptr, 4096, dist_buf_size)) {
            distances_buf = static_cast<float *>(aligned_ptr);
          } else {
            distances_buf = static_cast<float *>(alloc.alloc(dist_buf_size));
          }
          if (OB_ISNULL(distances_buf)) {
            metal_ok = false;
          } else {
            int gpu_ret = -1;
            common::VectorFormat fmt0 = vec0->get_format();
            if (fmt0 == common::VEC_FIXED) {
              common::ObFixedLengthBase *fix_base = static_cast<common::ObFixedLengthBase *>(vec0);
              const char *data_ptr = fix_base->get_data();
              const int64_t row_bytes = static_cast<int64_t>(fix_base->get_length());
              const int64_t batch_sz = bound.batch_size();
              if (OB_ISNULL(data_ptr) || batch_sz <= 0 || bound.end() > batch_sz) {
                metal_ok = false;
              } else if (row_bytes <= 0 || (elem_size == 4 && row_bytes != dim * 4) || (elem_size == 8 && row_bytes != dim * 8)) {
                metal_ok = false;
              } else {
                uint32_t *compact_offsets = static_cast<uint32_t *>(alloc.alloc(valid_cnt * sizeof(uint32_t)));
                if (OB_ISNULL(compact_offsets)) {
                  metal_ok = false;
                } else {
                  int64_t c = 0;
                  for (int64_t i = bound.start(); i < bound.end(); i++) {
                    if (skip.at(i) || eval_flags.at(i)) continue;
                    compact_offsets[c++] = static_cast<uint32_t>(i * row_bytes);
                  }
                  size_t data_byte_len = static_cast<size_t>(batch_sz * row_bytes);
                  if (elem_size == 8 && query_ptr_f64 != nullptr) {
                    gpu_ret = gpu_acc::vector_metal::metal_l2_query_vids_strided_f64(
                        query_ptr_f64, data_ptr, data_byte_len, compact_offsets, valid_cnt, dim, distances_buf);
                  } else if (elem_size == 4 && query_ptr != nullptr) {
                    gpu_ret = gpu_acc::vector_metal::metal_l2_query_vids_strided(
                        query_ptr, data_ptr, data_byte_len, compact_offsets, valid_cnt, dim, distances_buf);
                  } else {
                    metal_ok = false;
                  }
                }
              }
            } else if (fmt0 == common::VEC_CONTINUOUS && query_ptr != nullptr) {
              common::ObContinuousFormat *cont = static_cast<common::ObContinuousFormat *>(vec0);
              const char *data_ptr = cont->get_data();
              const uint32_t *offsets = cont->get_offsets();
              const int64_t batch_sz_cont = bound.batch_size();
              if (OB_ISNULL(data_ptr) || OB_ISNULL(offsets) || batch_sz_cont <= 0 || bound.end() > batch_sz_cont) {
                metal_ok = false;
              } else {
                uint32_t *compact_offsets = static_cast<uint32_t *>(alloc.alloc(valid_cnt * sizeof(uint32_t)));
                if (OB_ISNULL(compact_offsets)) {
                  metal_ok = false;
                } else {
                  int64_t c = 0;
                  for (int64_t i = bound.start(); i < bound.end(); i++) {
                    if (skip.at(i) || eval_flags.at(i)) continue;
                    compact_offsets[c++] = offsets[i];
                  }
                  size_t data_byte_len = static_cast<size_t>(offsets[batch_sz_cont]);
                  gpu_ret = gpu_acc::vector_metal::metal_l2_query_vids_strided(
                      query_ptr, data_ptr, data_byte_len, compact_offsets, valid_cnt, dim, distances_buf);
                }
              }
            }
            if (gpu_ret != 0 && metal_ok && query_ptr != nullptr) {
              const uint64_t vec_buf_size = static_cast<uint64_t>(valid_cnt) * static_cast<uint64_t>(dim) * sizeof(float);
              float *vectors_buf = static_cast<float *>(alloc.alloc(vec_buf_size));
              if (OB_ISNULL(vectors_buf)) {
                metal_ok = false;
              } else {
                int64_t compact = 0;
                for (int64_t i = bound.start(); OB_SUCC(ret) && compact < valid_cnt && i < bound.end(); i++) {
                  if (skip.at(i) || eval_flags.at(i)) continue;
                  const char *row_payload = nullptr;
                  int32_t row_len = 0;
                  if (get_compact_vector_payload(vec0, i, row_payload, row_len)
                      && (row_len / static_cast<int32_t>(sizeof(float)) == static_cast<int32_t>(dim))) {
                    memcpy(vectors_buf + compact * dim, row_payload, static_cast<size_t>(dim) * sizeof(float));
                  } else {
                    ObIArrayType *row_arr = nullptr;
                    bool row_null = false;
                    if (OB_FAIL(get_array_from_vector(*(expr.args_[0]), vec0, i, ctx, alloc, row_arr, row_null))) {
                      LOG_WARN("get_array_from_vector failed", K(ret), K(i));
                      metal_ok = false;
                      break;
                    }
                    if (row_null || OB_ISNULL(row_arr) || row_arr->size() != static_cast<uint32_t>(dim)) {
                      metal_ok = false;
                      break;
                    }
                    memcpy(vectors_buf + compact * dim, row_arr->get_data(), static_cast<size_t>(dim) * sizeof(float));
                  }
                  compact++;
                }
                if (OB_SUCC(ret) && metal_ok && compact >= static_cast<int64_t>(gpu_acc::vector_metal::METAL_L2_MIN_COUNT)) {
                  gpu_ret = gpu_acc::vector_metal::metal_l2_query_vids(
                      query_ptr, vectors_buf, compact, dim, distances_buf);
                }
              }
            }
            if (gpu_ret == 0) {
              int64_t dist_idx = 0;
              for (int64_t i = bound.start(); i < bound.end(); i++) {
                if (skip.at(i) || eval_flags.at(i)) continue;
                res_vec->set_float(i, static_cast<float>(sqrt(static_cast<double>(distances_buf[dist_idx++]))));
                eval_flags.set(i);
              }
            } else {
              metal_ok = false;
            }
            if (aligned_ptr != nullptr) {
              free(aligned_ptr);
            }
          }
        }
        if (!metal_ok || OB_FAIL(ret)) {
          ret = l2_vector_cpu_loop(expr, ctx, skip, bound, res_vec, vec0, vec1, alloc);
        }
      }
    }
  }
  return ret;
}
#else
int ObExprVectorL2Distance::calc_l2_distance_vector_metal(VECTOR_EVAL_FUNC_ARG_DECL)
{
  return calc_l2_distance_vector_cpu(expr, ctx, skip, bound);
}
#endif

// -------- ObExprVectorCosineDistance batch / vector (CPU + Metal) --------
static int cosine_batch_cpu_loop(const ObExpr &expr, ObEvalCtx &ctx, const ObBitVector &skip,
                                 const int64_t batch_size, ObDatum *results, ObBitVector &eval_flags)
{
  int ret = OB_SUCCESS;
  for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; i++) {
    if (skip.at(i) || eval_flags.at(i)) continue;
    ObDatum &res = results[i];
    if (OB_FAIL(ObExprVectorDistance::calc_distance(expr, ctx, res, ObExprVectorDistance::ObVecDisType::COSINE))) {
      LOG_WARN("calc_distance failed", K(ret), K(i));
    } else {
      eval_flags.set(i);
    }
  }
  return ret;
}

int ObExprVectorCosineDistance::calc_cosine_distance_batch_cpu(const ObExpr &expr, ObEvalCtx &ctx,
                                                               const ObBitVector &skip, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(expr.args_[0]->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval batch arg0 failed", K(ret));
  } else if (OB_FAIL(expr.args_[1]->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval batch arg1 failed", K(ret));
  } else {
    ObDatum *results = expr.locate_batch_datums(ctx);
    ObDatum *d0 = expr.args_[0]->locate_batch_datums(ctx);
    if (OB_ISNULL(results) || OB_ISNULL(d0)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("batch datum is null", K(ret));
    } else {
      ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
      ret = cosine_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
    }
  }
  return ret;
}

#ifdef __APPLE__
int ObExprVectorCosineDistance::calc_cosine_distance_batch_metal(const ObExpr &expr, ObEvalCtx &ctx,
                                                                 const ObBitVector &skip, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(expr.args_[0]->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval batch arg0 failed", K(ret));
  } else if (OB_FAIL(expr.args_[1]->eval_batch(ctx, skip, batch_size))) {
    LOG_WARN("eval batch arg1 failed", K(ret));
  } else {
    ObDatum *results = expr.locate_batch_datums(ctx);
    ObDatum *d0 = expr.args_[0]->locate_batch_datums(ctx);
    ObDatum *d1 = expr.args_[1]->locate_batch_datums(ctx);
    if (OB_ISNULL(results) || OB_ISNULL(d0) || OB_ISNULL(d1)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("batch datum is null", K(ret));
    } else {
      ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
      int64_t valid_cnt = 0;
      for (int64_t i = 0; i < batch_size; i++) {
        if (!skip.at(i) && !eval_flags.at(i)) valid_cnt++;
      }
      bool do_metal = (valid_cnt >= static_cast<int64_t>(gpu_acc::vector_metal::METAL_COSINE_MIN_COUNT));
      if (!do_metal || valid_cnt <= 0) {
        ret = cosine_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
      } else {
        ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
        common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();
        ObIArrayType *query_arr = nullptr;
        bool contain_null = false;
        if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[1]), ctx, alloc, query_arr, contain_null))
            || OB_ISNULL(query_arr) || contain_null || query_arr->size() == 0) {
          ret = cosine_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
        } else {
          const int64_t dim = static_cast<int64_t>(query_arr->size());
          const float *query_ptr = reinterpret_cast<const float *>(query_arr->get_data());
          const uint64_t vec_buf_size = static_cast<uint64_t>(valid_cnt) * static_cast<uint64_t>(dim) * sizeof(float);
          const uint64_t dist_buf_size = static_cast<uint64_t>(valid_cnt) * sizeof(float);
          float *vectors_buf = static_cast<float *>(alloc.alloc(vec_buf_size));
          void *batch_aligned_ptr = nullptr;
          float *distances_buf = nullptr;
          if (0 == posix_memalign(&batch_aligned_ptr, 4096, dist_buf_size)) {
            distances_buf = static_cast<float *>(batch_aligned_ptr);
          } else {
            distances_buf = static_cast<float *>(alloc.alloc(dist_buf_size));
          }
          if (OB_ISNULL(vectors_buf) || OB_ISNULL(distances_buf)) {
            ret = cosine_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
          } else {
            int64_t compact = 0;
            for (int64_t i = 0; OB_SUCC(ret) && compact < valid_cnt && i < batch_size; i++) {
              if (skip.at(i) || eval_flags.at(i)) continue;
              if (d0[i].is_null()) {
                results[i].set_null();
                eval_flags.set(i);
                continue;
              }
              ObIArrayType *row_arr = nullptr;
              if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[0]), d0[i], ctx, alloc, row_arr))) {
                LOG_WARN("get_type_vector failed", K(ret), K(i));
                break;
              }
              if (OB_ISNULL(row_arr) || row_arr->size() != query_arr->size()) {
                ret = OB_ERR_UNEXPECTED;
                LOG_WARN("vector dim mismatch", K(ret), K(i));
                break;
              }
              memcpy(vectors_buf + compact * dim, row_arr->get_data(), static_cast<size_t>(dim) * sizeof(float));
              compact++;
            }
            if (OB_SUCC(ret) && compact >= static_cast<int64_t>(gpu_acc::vector_metal::METAL_COSINE_MIN_COUNT)) {
              int gpu_ret = gpu_acc::vector_metal::metal_cosine_query_vids(
                  query_ptr, vectors_buf, compact, dim, distances_buf);
              if (gpu_ret == 0) {
                compact = 0;
                for (int64_t i = 0; i < batch_size; i++) {
                  if (skip.at(i) || eval_flags.at(i)) continue;
                  if (d0[i].is_null()) continue;
                  results[i].set_float(distances_buf[compact]);
                  eval_flags.set(i);
                  compact++;
                }
              } else {
                ret = cosine_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
              }
            } else {
              ret = cosine_batch_cpu_loop(expr, ctx, skip, batch_size, results, eval_flags);
            }
            if (batch_aligned_ptr != nullptr) {
              free(batch_aligned_ptr);
            }
          }
        }
      }
    }
  }
  return ret;
}
#else
int ObExprVectorCosineDistance::calc_cosine_distance_batch_metal(const ObExpr &expr, ObEvalCtx &ctx,
                                                                  const ObBitVector &skip, const int64_t batch_size)
{
  return calc_cosine_distance_batch_cpu(expr, ctx, skip, batch_size);
}
#endif

static int cosine_vector_cpu_loop(const ObExpr &expr, ObEvalCtx &ctx, const ObBitVector &skip,
                                  const EvalBound &bound, common::ObIVector *res_vec,
                                  common::ObIVector *vec0, common::ObIVector *vec1,
                                  common::ObArenaAllocator &alloc)
{
  int ret = OB_SUCCESS;
  ObIArrayType *query_arr = nullptr;
  bool query_null = false;
  if (OB_FAIL(get_array_from_vector(*(expr.args_[1]), vec1, bound.start(), ctx, alloc, query_arr, query_null))) {
    LOG_WARN("get query array from vector failed", K(ret));
  } else if (query_null || OB_ISNULL(query_arr) || query_arr->size() == 0) {
    ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
    for (int64_t i = bound.start(); OB_SUCC(ret) && i < bound.end(); i++) {
      if (skip.at(i) || eval_flags.at(i)) continue;
      res_vec->set_null(i);
      eval_flags.set(i);
    }
  } else {
    const float *query_data = reinterpret_cast<const float *>(query_arr->get_data());
    const uint32_t dim = query_arr->size();
    ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
    for (int64_t i = bound.start(); OB_SUCC(ret) && i < bound.end(); i++) {
      if (skip.at(i) || eval_flags.at(i)) continue;
      ObIArrayType *row_arr = nullptr;
      bool row_null = false;
      if (OB_FAIL(get_array_from_vector(*(expr.args_[0]), vec0, i, ctx, alloc, row_arr, row_null))) {
        LOG_WARN("get_array_from_vector failed", K(ret), K(i));
      } else if (row_null || OB_ISNULL(row_arr)) {
        res_vec->set_null(i);
        eval_flags.set(i);
      } else if (row_arr->size() != dim) {
        ret = OB_ERR_INVALID_VECTOR_DIM;
        LOG_WARN("vector dim mismatch", K(ret), K(row_arr->size()), K(dim));
      } else if (row_arr->contain_null() || query_arr->contain_null()) {
        res_vec->set_null(i);
        eval_flags.set(i);
      } else {
        double distance = 0.0;
        const float *row_data = reinterpret_cast<const float *>(row_arr->get_data());
        if (ObExprVectorDistance::DisFunc<float>::distance_funcs[ObExprVectorDistance::ObVecDisType::COSINE] == nullptr) {
          ret = OB_NOT_SUPPORTED;
          LOG_WARN("cosine distance func not supported", K(ret));
        } else if (OB_FAIL(ObExprVectorDistance::DisFunc<float>::distance_funcs[ObExprVectorDistance::ObVecDisType::COSINE](row_data, query_data, dim, distance))) {
          if (ret == OB_ERR_NULL_VALUE) {
            res_vec->set_null(i);
            eval_flags.set(i);
            ret = OB_SUCCESS;
          } else {
            LOG_WARN("calc cosine distance failed", K(ret), K(i));
          }
        } else {
          res_vec->set_float(i, static_cast<float>(distance));
          eval_flags.set(i);
        }
      }
    }
  }
  return ret;
}

int ObExprVectorCosineDistance::calc_cosine_distance_vector_cpu(VECTOR_EVAL_FUNC_ARG_DECL)
{
  int ret = OB_SUCCESS;
  const int64_t batch_size = bound.batch_size();
  if (OB_FAIL(expr.args_[0]->eval_vector(ctx, skip, bound))) {
    LOG_WARN("eval_vector arg0 failed", K(ret));
  } else if (OB_FAIL(expr.args_[1]->eval_vector(ctx, skip, bound))) {
    LOG_WARN("eval_vector arg1 failed", K(ret));
  } else if (OB_FAIL(expr.init_vector_for_write(ctx, VEC_UNIFORM, batch_size))) {
    LOG_WARN("init_vector_for_write failed", K(ret));
  } else {
    common::ObIVector *res_vec = expr.get_vector(ctx);
    common::ObIVector *vec0 = expr.args_[0]->get_vector(ctx);
    common::ObIVector *vec1 = expr.args_[1]->get_vector(ctx);
    if (OB_ISNULL(res_vec) || OB_ISNULL(vec0) || OB_ISNULL(vec1)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("get_vector returned null", K(ret));
    } else {
      ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
      common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();
      ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
      batch_guard.set_batch_size(batch_size);
      ret = cosine_vector_cpu_loop(expr, ctx, skip, bound, res_vec, vec0, vec1, alloc);
    }
  }
  return ret;
}

#ifdef __APPLE__
int ObExprVectorCosineDistance::calc_cosine_distance_vector_metal(VECTOR_EVAL_FUNC_ARG_DECL)
{
  int ret = OB_SUCCESS;
  const int64_t batch_size = bound.batch_size();
  if (OB_FAIL(expr.args_[0]->eval_vector(ctx, skip, bound))) {
    LOG_WARN("eval_vector arg0 failed", K(ret));
  } else if (OB_FAIL(expr.args_[1]->eval_vector(ctx, skip, bound))) {
    LOG_WARN("eval_vector arg1 failed", K(ret));
  } else if (OB_FAIL(expr.init_vector_for_write(ctx, VEC_UNIFORM, batch_size))) {
    LOG_WARN("init_vector_for_write failed", K(ret));
  } else {
    common::ObIVector *res_vec = expr.get_vector(ctx);
    common::ObIVector *vec0 = expr.args_[0]->get_vector(ctx);
    common::ObIVector *vec1 = expr.args_[1]->get_vector(ctx);
    if (OB_ISNULL(res_vec) || OB_ISNULL(vec0) || OB_ISNULL(vec1)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("get_vector returned null", K(ret));
    } else {
      ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
      ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
      common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();
      ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
      batch_guard.set_batch_size(batch_size);

      int64_t valid_cnt = 0;
      for (int64_t i = bound.start(); i < bound.end(); i++) {
        if (!skip.at(i) && !eval_flags.at(i)) valid_cnt++;
      }
      bool do_metal = (valid_cnt >= static_cast<int64_t>(gpu_acc::vector_metal::METAL_COSINE_MIN_COUNT));
      if (!do_metal || valid_cnt <= 0) {
        ret = cosine_vector_cpu_loop(expr, ctx, skip, bound, res_vec, vec0, vec1, alloc);
      } else {
        bool metal_ok = true;
        const float *query_ptr = nullptr;
        const double *query_ptr_f64 = nullptr;
        int64_t dim = 0;
        int elem_size = 4;
        const char *query_payload = nullptr;
        int32_t query_payload_len = 0;
        bool query_compact = get_compact_vector_payload(vec1, bound.start(), query_payload, query_payload_len);
        if (query_compact && query_payload_len > 0) {
          common::VectorFormat fmt0 = vec0->get_format();
          if (fmt0 == common::VEC_FIXED) {
            common::ObFixedLengthBase *fix_base = static_cast<common::ObFixedLengthBase *>(vec0);
            const int64_t row_bytes = static_cast<int64_t>(fix_base->get_length());
            if (row_bytes > 0 && query_payload_len == row_bytes) {
              if (row_bytes % sizeof(float) == 0) {
                dim = row_bytes / static_cast<int64_t>(sizeof(float));
                query_ptr = reinterpret_cast<const float *>(query_payload);
                elem_size = 4;
              } else if (row_bytes % sizeof(double) == 0) {
                dim = row_bytes / static_cast<int64_t>(sizeof(double));
                query_ptr_f64 = reinterpret_cast<const double *>(query_payload);
                elem_size = 8;
              }
            }
          }
          if (dim <= 0) {
            dim = static_cast<int64_t>(query_payload_len) / static_cast<int64_t>(sizeof(float));
            if (dim > 0 && static_cast<int64_t>(query_payload_len) == dim * static_cast<int64_t>(sizeof(float))) {
              query_ptr = reinterpret_cast<const float *>(query_payload);
              elem_size = 4;
            } else {
              dim = static_cast<int64_t>(query_payload_len) / static_cast<int64_t>(sizeof(double));
              if (dim > 0 && static_cast<int64_t>(query_payload_len) == dim * static_cast<int64_t>(sizeof(double))) {
                query_ptr_f64 = reinterpret_cast<const double *>(query_payload);
                elem_size = 8;
              } else {
                query_ptr = reinterpret_cast<const float *>(query_payload);
                dim = static_cast<int64_t>(query_payload_len) / static_cast<int64_t>(sizeof(float));
                elem_size = 4;
              }
            }
          }
        }
        if (query_ptr == nullptr && query_ptr_f64 == nullptr) {
          ObIArrayType *query_arr = nullptr;
          bool query_null = false;
          if (OB_FAIL(get_array_from_vector(*(expr.args_[1]), vec1, bound.start(), ctx, alloc, query_arr, query_null))
              || query_null || OB_ISNULL(query_arr) || query_arr->size() == 0) {
            metal_ok = false;
          } else {
            dim = static_cast<int64_t>(query_arr->size());
            query_ptr = reinterpret_cast<const float *>(query_arr->get_data());
            elem_size = 4;
          }
        }
        if (metal_ok && dim > 0 && (query_ptr != nullptr || query_ptr_f64 != nullptr)) {
          const uint64_t dist_buf_size = static_cast<uint64_t>(valid_cnt) * sizeof(float);
          float *distances_buf = nullptr;
          void *aligned_ptr = nullptr;
          if (0 == posix_memalign(&aligned_ptr, 4096, dist_buf_size)) {
            distances_buf = static_cast<float *>(aligned_ptr);
          } else {
            distances_buf = static_cast<float *>(alloc.alloc(dist_buf_size));
          }
          if (OB_ISNULL(distances_buf)) {
            metal_ok = false;
          } else {
            int gpu_ret = -1;
            common::VectorFormat fmt0 = vec0->get_format();
            if (fmt0 == common::VEC_FIXED) {
              common::ObFixedLengthBase *fix_base = static_cast<common::ObFixedLengthBase *>(vec0);
              const char *data_ptr = fix_base->get_data();
              const int64_t row_bytes = static_cast<int64_t>(fix_base->get_length());
              const int64_t batch_sz = bound.batch_size();
              if (OB_ISNULL(data_ptr) || batch_sz <= 0 || bound.end() > batch_sz) {
                metal_ok = false;
              } else if (row_bytes <= 0 || (elem_size == 4 && row_bytes != dim * 4) || (elem_size == 8 && row_bytes != dim * 8)) {
                metal_ok = false;
              } else {
                uint32_t *compact_offsets = static_cast<uint32_t *>(alloc.alloc(valid_cnt * sizeof(uint32_t)));
                if (OB_ISNULL(compact_offsets)) {
                  metal_ok = false;
                } else {
                  int64_t c = 0;
                  for (int64_t i = bound.start(); i < bound.end(); i++) {
                    if (skip.at(i) || eval_flags.at(i)) continue;
                    compact_offsets[c++] = static_cast<uint32_t>(i * row_bytes);
                  }
                  size_t data_byte_len = static_cast<size_t>(batch_sz * row_bytes);
                  if (elem_size == 8 && query_ptr_f64 != nullptr) {
                    gpu_ret = gpu_acc::vector_metal::metal_cosine_query_vids_strided_f64(
                        query_ptr_f64, data_ptr, data_byte_len, compact_offsets, valid_cnt, dim, distances_buf);
                  } else if (elem_size == 4 && query_ptr != nullptr) {
                    gpu_ret = gpu_acc::vector_metal::metal_cosine_query_vids_strided(
                        query_ptr, data_ptr, data_byte_len, compact_offsets, valid_cnt, dim, distances_buf);
                  } else {
                    metal_ok = false;
                  }
                }
              }
            } else if (fmt0 == common::VEC_CONTINUOUS && query_ptr != nullptr) {
              common::ObContinuousFormat *cont = static_cast<common::ObContinuousFormat *>(vec0);
              const char *data_ptr = cont->get_data();
              const uint32_t *offsets = cont->get_offsets();
              const int64_t batch_sz_cont = bound.batch_size();
              if (OB_ISNULL(data_ptr) || OB_ISNULL(offsets) || batch_sz_cont <= 0 || bound.end() > batch_sz_cont) {
                metal_ok = false;
              } else {
                uint32_t *compact_offsets = static_cast<uint32_t *>(alloc.alloc(valid_cnt * sizeof(uint32_t)));
                if (OB_ISNULL(compact_offsets)) {
                  metal_ok = false;
                } else {
                  int64_t c = 0;
                  for (int64_t i = bound.start(); i < bound.end(); i++) {
                    if (skip.at(i) || eval_flags.at(i)) continue;
                    compact_offsets[c++] = offsets[i];
                  }
                  size_t data_byte_len = static_cast<size_t>(offsets[batch_sz_cont]);
                  gpu_ret = gpu_acc::vector_metal::metal_cosine_query_vids_strided(
                      query_ptr, data_ptr, data_byte_len, compact_offsets, valid_cnt, dim, distances_buf);
                }
              }
            }
            if (gpu_ret != 0 && metal_ok && query_ptr != nullptr) {
              const uint64_t vec_buf_size = static_cast<uint64_t>(valid_cnt) * static_cast<uint64_t>(dim) * sizeof(float);
              float *vectors_buf = static_cast<float *>(alloc.alloc(vec_buf_size));
              if (OB_ISNULL(vectors_buf)) {
                metal_ok = false;
              } else {
                int64_t compact = 0;
                for (int64_t i = bound.start(); OB_SUCC(ret) && compact < valid_cnt && i < bound.end(); i++) {
                  if (skip.at(i) || eval_flags.at(i)) continue;
                  const char *row_payload = nullptr;
                  int32_t row_len = 0;
                  if (get_compact_vector_payload(vec0, i, row_payload, row_len)
                      && (row_len / static_cast<int32_t>(sizeof(float)) == static_cast<int32_t>(dim))) {
                    memcpy(vectors_buf + compact * dim, row_payload, static_cast<size_t>(dim) * sizeof(float));
                  } else {
                    ObIArrayType *row_arr = nullptr;
                    bool row_null = false;
                    if (OB_FAIL(get_array_from_vector(*(expr.args_[0]), vec0, i, ctx, alloc, row_arr, row_null))) {
                      LOG_WARN("get_array_from_vector failed", K(ret), K(i));
                      metal_ok = false;
                      break;
                    }
                    if (row_null || OB_ISNULL(row_arr) || row_arr->size() != static_cast<uint32_t>(dim)) {
                      metal_ok = false;
                      break;
                    }
                    memcpy(vectors_buf + compact * dim, row_arr->get_data(), static_cast<size_t>(dim) * sizeof(float));
                  }
                  compact++;
                }
                if (OB_SUCC(ret) && metal_ok && compact >= static_cast<int64_t>(gpu_acc::vector_metal::METAL_COSINE_MIN_COUNT)) {
                  gpu_ret = gpu_acc::vector_metal::metal_cosine_query_vids(
                      query_ptr, vectors_buf, compact, dim, distances_buf);
                }
              }
            }
            if (gpu_ret == 0) {
              int64_t dist_idx = 0;
              for (int64_t i = bound.start(); i < bound.end(); i++) {
                if (skip.at(i) || eval_flags.at(i)) continue;
                res_vec->set_float(i, distances_buf[dist_idx++]);
                eval_flags.set(i);
              }
            } else {
              metal_ok = false;
            }
            if (aligned_ptr != nullptr) {
              free(aligned_ptr);
            }
          }
        }
        if (!metal_ok || OB_FAIL(ret)) {
          ret = cosine_vector_cpu_loop(expr, ctx, skip, bound, res_vec, vec0, vec1, alloc);
        }
      }
    }
  }
  return ret;
}
#else
int ObExprVectorCosineDistance::calc_cosine_distance_vector_metal(VECTOR_EVAL_FUNC_ARG_DECL)
{
  return calc_cosine_distance_vector_cpu(expr, ctx, skip, bound);
}
#endif

ObExprVectorL2Squared::ObExprVectorL2Squared(ObIAllocator &alloc)
    : ObExprVectorDistance(alloc, T_FUN_SYS_L2_SQUARED, N_VECTOR_L2_SQUARED, 2, NOT_ROW_DIMENSION) {}

int ObExprVectorL2Squared::calc_result_type2(
    ObExprResType &type,
    ObExprResType &type1,
    ObExprResType &type2,
    common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;
  uint16_t unused_id = UINT16_MAX;
  if (OB_FAIL(ObArrayExprUtils::calc_cast_type2(type_, type1, type2, type_ctx, unused_id))) {
    LOG_WARN("failed to calc cast type", K(ret), K(type1));
  } else {
    type.set_type(ObFloatType);
    type.set_calc_type(ObFloatType);
  }
  return ret;
}

int ObExprVectorL2Squared::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                    ObExpr &rt_expr) const
{
    int ret = OB_SUCCESS;  
    rt_expr.eval_func_ = ObExprVectorL2Squared::calc_l2_squared;
    return ret;
}

int ObExprVectorL2Squared::calc_l2_squared(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  return ObExprVectorDistance::calc_distance(expr, ctx, res_datum, ObVecDisType::EUCLIDEAN_SQUARED);
}

ObExprVectorCosineDistance::ObExprVectorCosineDistance(ObIAllocator &alloc)
    : ObExprVectorDistance(alloc, T_FUN_SYS_COSINE_DISTANCE, N_VECTOR_COS_DISTANCE, 2, NOT_ROW_DIMENSION) {}

int ObExprVectorCosineDistance::calc_result_type2(
    ObExprResType &type,
    ObExprResType &type1,
    ObExprResType &type2,
    common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;
  uint16_t unused_id = UINT16_MAX;
  if (OB_FAIL(ObArrayExprUtils::calc_cast_type2(type_, type1, type2, type_ctx, unused_id))) {
    LOG_WARN("failed to calc cast type", K(ret), K(type1));
  } else {
    type.set_type(ObFloatType);
    type.set_calc_type(ObFloatType);
  }
  return ret;
}

int ObExprVectorCosineDistance::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                    ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  rt_expr.eval_func_ = ObExprVectorCosineDistance::calc_cosine_distance;
  ObExprCosineMetalExtraInfo *extra = nullptr;
  if (OB_FAIL(ObExprExtraInfoFactory::alloc(*expr_cg_ctx.allocator_, T_FUN_SYS_COSINE_DISTANCE, reinterpret_cast<ObIExprExtraInfo *&>(extra)))) {
    LOG_WARN("alloc Cosine metal extra info failed", K(ret));
  } else {
#ifdef __APPLE__
    extra->use_metal_ = gpu_acc::vector_metal::is_metal_cosine_ready();
#else
    extra->use_metal_ = false;
#endif
    rt_expr.extra_info_ = extra;
    if (extra->use_metal_) {
      rt_expr.eval_batch_func_ = ObExprVectorCosineDistance::calc_cosine_distance_batch_metal;
      rt_expr.eval_vector_func_ = ObExprVectorCosineDistance::calc_cosine_distance_vector_metal;
    } else {
      rt_expr.eval_batch_func_ = ObExprVectorCosineDistance::calc_cosine_distance_batch_cpu;
      rt_expr.eval_vector_func_ = ObExprVectorCosineDistance::calc_cosine_distance_vector_cpu;
    }
  }
  return ret;
}

int ObExprVectorCosineDistance::calc_cosine_distance(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  return ObExprVectorDistance::calc_distance(expr, ctx, res_datum, ObVecDisType::COSINE);
}

ObExprVectorIPDistance::ObExprVectorIPDistance(ObIAllocator &alloc)
    : ObExprVectorDistance(alloc, T_FUN_SYS_INNER_PRODUCT, N_VECTOR_INNER_PRODUCT, 2, NOT_ROW_DIMENSION) {}

int ObExprVectorIPDistance::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                    ObExpr &rt_expr) const
{
    int ret = OB_SUCCESS;  
    rt_expr.eval_func_ = ObExprVectorIPDistance::calc_inner_product;
    return ret;
}

int ObExprVectorIPDistance::calc_inner_product(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  return ObExprVectorDistance::calc_distance(expr, ctx, res_datum, ObVecDisType::DOT);
}

ObExprVectorNegativeIPDistance::ObExprVectorNegativeIPDistance(ObIAllocator &alloc)
    : ObExprVectorDistance(alloc, T_FUN_SYS_NEGATIVE_INNER_PRODUCT, N_VECTOR_NEGATIVE_INNER_PRODUCT, 2, NOT_ROW_DIMENSION) {}

int ObExprVectorNegativeIPDistance::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                    ObExpr &rt_expr) const
{
    int ret = OB_SUCCESS;  
    rt_expr.eval_func_ = ObExprVectorNegativeIPDistance::calc_negative_inner_product;
    return ret;
}

int ObExprVectorNegativeIPDistance::calc_negative_inner_product(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObExprVectorDistance::calc_distance(expr, ctx, res_datum, ObVecDisType::DOT))) {
    LOG_WARN("fail to calc distance", K(ret), K(ObVecDisType::DOT));
  } else if (!res_datum.is_null() && res_datum.get_double() != 0) {
    double value = -1 * res_datum.get_double();
    res_datum.set_double(value);
  }
  return ret;
}

ObExprVectorDims::ObExprVectorDims(ObIAllocator &alloc)
    : ObExprVector(alloc, T_FUN_SYS_VECTOR_DIMS, N_VECTOR_DIMS, 1, NOT_ROW_DIMENSION) {}

int ObExprVectorDims::calc_result_type1(
    ObExprResType &type,
    ObExprResType &type1,
    common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS; 
  if (OB_FAIL(ObArrayExprUtils::calc_cast_type(type_, type1, type_ctx))) {
    LOG_WARN("failed to calc cast type", K(ret), K(type1));
  } else {
    type.set_type(ObIntType);
    type.set_precision(ObAccuracy::DDL_DEFAULT_ACCURACY[ObIntType].precision_);
    type.set_scale(ObAccuracy::DDL_DEFAULT_ACCURACY[ObIntType].scale_);
    type.set_calc_type(ObIntType);
  }
  return ret;
}
int ObExprVectorDims::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                    ObExpr &rt_expr) const
{
    int ret = OB_SUCCESS;
    rt_expr.eval_func_ = ObExprVectorDims::calc_dims;
    if (rt_expr.arg_cnt_ != 1 || OB_ISNULL(rt_expr.args_)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("count of children is not 1 or children is null", K(ret), K(rt_expr.arg_cnt_), K(rt_expr.args_));
    } else if (rt_expr.args_[0]->type_ == T_FUN_SYS_CAST) {
      // return error if cast failed
      rt_expr.args_[0]->extra_  &= ~CM_WARN_ON_FAIL;
    }
    return ret;
}

int ObExprVectorDims::calc_dims(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  int ret = OB_SUCCESS;
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &tmp_allocator = tmp_alloc_g.get_allocator();
  ObIArrayType *arr = NULL;
  bool contain_null = false;
  if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[0]), ctx, tmp_allocator, arr, contain_null))) {
    LOG_WARN("failed to get vector", K(ret), K(*expr.args_[0]));
  } else if (contain_null) {
    res_datum.set_null();
  } else if (OB_ISNULL(arr)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected nullptr", K(ret), K(arr));
  } else if (arr->contain_null()) {
    ret = OB_ERR_NULL_VALUE;
    LOG_WARN("array with null can't calculate vector norm", K(ret));
  } else {
    res_datum.set_int(arr->size());
  }
  return ret;
}

ObExprVectorNorm::ObExprVectorNorm(ObIAllocator &alloc)
    : ObExprVector(alloc, T_FUN_SYS_VECTOR_NORM, N_VECTOR_NORM, 1, NOT_ROW_DIMENSION) {}

int ObExprVectorNorm::cg_expr(ObExprCGCtx &expr_cg_ctx, const ObRawExpr &raw_expr,
                                    ObExpr &rt_expr) const
{
    int ret = OB_SUCCESS;  
    rt_expr.eval_func_ = ObExprVectorNorm::calc_norm;
    if (rt_expr.arg_cnt_ != 1 || OB_ISNULL(rt_expr.args_)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("count of children is not 1 or children is null", K(ret), K(rt_expr.arg_cnt_), K(rt_expr.args_));
    } else if (rt_expr.args_[0]->type_ == T_FUN_SYS_CAST) {
      // return error if cast failed
      rt_expr.args_[0]->extra_  &= ~CM_WARN_ON_FAIL;
    }
    return ret;
}

int ObExprVectorNorm::calc_norm(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res_datum)
{
  int ret = OB_SUCCESS;
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &tmp_allocator = tmp_alloc_g.get_allocator();
  ObIArrayType *arr = NULL;
  bool contain_null = false;
  if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[0]), ctx, tmp_allocator, arr, contain_null))) {
    LOG_WARN("failed to get vector", K(ret), K(*expr.args_[0]));
  } else if (contain_null) {
    res_datum.set_null();
  } else if (OB_ISNULL(arr)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected nullptr", K(ret), K(arr));
  } else if (arr->contain_null()) {
    ret = OB_ERR_NULL_VALUE;
    LOG_WARN("array with null can't calculate vector norm", K(ret));
  } else {
    double norm = 0.0;
    const float *data = reinterpret_cast<const float*>(arr->get_data());
    if (OB_FAIL(ObVectorNorm::vector_norm_func(data, arr->size(), norm))) {
      LOG_WARN("failed to calc vector norm", K(ret));
    } else {
      res_datum.set_double(norm);
    }
  }
  return ret;
}

} // sql
} // oceanbase
