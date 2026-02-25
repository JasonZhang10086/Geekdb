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

#include "sql/engine/expr/ob_expr_vec_ivf_center_id.h"
#include "sql/engine/expr/ob_expr_calc_partition_id.h"
#include "sql/engine/expr/ob_array_expr_utils.h"
#include "sql/engine/expr/ob_expr.h"
#include "share/vector_index/ob_vector_index_util.h"
#include "share/vector_index/ob_plugin_vector_index_service.h"
#include "share/vector_type/ob_vector_common_util.h"
#ifdef __APPLE__
#include "gpu_acc/metal/vector/ob_metal_l2.h"
#include <cmath>
#endif

namespace oceanbase
{
using namespace common;
namespace sql
{

ObExprVecIVFCenterID::ObExprVecIVFCenterID(ObIAllocator &allocator)
  : ObFuncExprOperator(allocator, T_FUN_SYS_VEC_IVF_CENTER_ID, N_VEC_IVF_CENTER_ID, MORE_THAN_ZERO, VALID_FOR_GENERATED_COL, NOT_ROW_DIMENSION)
{
  need_charset_convert_ = false;
}

int ObExprVecIVFCenterID::calc_result_typeN(ObExprResType &type,
                                       ObExprResType *types,
                                       int64_t param_num,
                                       ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;
  UNUSEDx(param_num, types);
  type.set_varchar();
  type.set_collation_type(CS_TYPE_BINARY);
  return ret;
}

int ObExprVecIVFCenterID::calc_resultN(ObObj &result,
                                  const ObObj *objs_array,
                                  int64_t param_num,
                                  ObExprCtx &expr_ctx) const
{
  // TODO by query ivf index
  return OB_NOT_SUPPORTED;
}

int ObExprVecIVFCenterID::cg_expr(
    ObExprCGCtx &expr_cg_ctx,
    const ObRawExpr &raw_expr,
    ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(rt_expr.arg_cnt_ != 4 && rt_expr.arg_cnt_ != 1 && rt_expr.arg_cnt_ != 2)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected param count", K(rt_expr.arg_cnt_), K(rt_expr.args_), K(rt_expr.type_));
  } else if (OB_UNLIKELY(rt_expr.arg_cnt_ == 4 && OB_ISNULL(rt_expr.args_))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected error, rt_expr.args_ is nullptr", K(rt_expr.arg_cnt_), K(rt_expr.args_), K(rt_expr.type_));
  } else {
    rt_expr.eval_func_ = calc_center_id;
    // Always set batch/vector so we never fall back to expr_default_eval_batch_func.
    // Use Metal for all arg_cnt_ (1/2/4) when available; Metal entry will fallback to CPU when arg_cnt_!=4 at runtime.
    // This ensures the same plan (e.g. arg_cnt_=1 at CG) still gets Metal pointers so execution can use GPU when applicable.
#ifdef __APPLE__
    bool metal_ready = gpu_acc::vector_metal::is_metal_ready();
    if (metal_ready) {
      rt_expr.eval_batch_func_ = calc_center_id_batch_metal;
      rt_expr.eval_vector_func_ = calc_center_id_vector_metal;
    } else
#endif
    {
      rt_expr.eval_batch_func_ = calc_center_id_batch_cpu;
      rt_expr.eval_vector_func_ = calc_center_id_vector_cpu;
    }
  }
  return ret;
}

int ObExprVecIVFCenterID::calc_center_id(
    const ObExpr &expr,
    ObEvalCtx &eval_ctx,
    ObDatum &expr_datum)
{
  int ret = OB_SUCCESS;
  if (expr.arg_cnt_ == 1) {
    expr_datum.set_null();
  } else if (expr.arg_cnt_ == 2) {
    int64_t buf_len = OB_DOC_ID_COLUMN_BYTE_LENGTH;
    char *buf = expr.get_str_res_mem(eval_ctx, buf_len);
    ObString str(buf_len, 0, buf);
    ObCenterId center_id(1, 0);
    if (OB_FAIL(ObVectorClusterHelper::set_center_id_to_string(center_id, str))) {
      LOG_WARN("failed to set center_id to string", K(ret), K(center_id), K(str));
    } else {
      expr_datum.set_string(str);
    }
  } else if (OB_UNLIKELY(4 != expr.arg_cnt_) || OB_ISNULL(expr.args_)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid arguments", K(ret), K(expr), KP(expr.args_));
  } else {
    common::ObArenaAllocator tmp_allocator("IVFExprCID", OB_MALLOC_NORMAL_BLOCK_SIZE, MTL_ID());
    ObTableID table_id;
    ObTabletID tablet_id;
    ObVectorIndexDistAlgorithm dis_algo = VIDA_MAX;
    ObSEArray<float*, 64> centers;
    bool contain_null = false;
    ObIArrayType *arr = NULL;
    int64_t center_idx = 0; // use 0 as center idx if vector is null
    uint64_t center_prefix = 0;
    if (OB_FAIL(ObVectorIndexUtil::eval_ivf_centers_common(
        tmp_allocator, expr, eval_ctx, centers, table_id, tablet_id, dis_algo, contain_null, arr, center_prefix))) {
      LOG_WARN("failed to eval ivf centers", K(ret), K(expr), K(eval_ctx));
    } else if (contain_null) {
      // do nothing
    } else {
      ObVectorClusterHelper helper;
      ObVectorNormalizeInfo norm_info;
      if (OB_FAIL(helper.get_nearest_probe_centers(
          reinterpret_cast<float*>(arr->get_data()),
          arr->size(),
          centers,
          1/*nprobe*/,
          tmp_allocator,
          VIDA_COS != dis_algo ? nullptr: &norm_info))) {
        LOG_WARN("failed to get nearest center", K(ret));
      } else if (OB_FAIL(helper.get_center_idx(0, center_idx))) {
        LOG_WARN("failed to get center idx", K(ret));
      }
    }
    if (OB_SUCC(ret)) {
      int64_t buf_len = OB_DOC_ID_COLUMN_BYTE_LENGTH;
      char *buf = expr.get_str_res_mem(eval_ctx, buf_len);
      ObString str(buf_len, 0, buf);
      ObCenterId center_id(center_prefix, center_idx);
      if (OB_FAIL(ObVectorClusterHelper::set_center_id_to_string(center_id, str))) {
        LOG_WARN("failed to set center_id to string", K(ret), K(center_id), K(str));
      } else {
        expr_datum.set_string(str);
      }
    }
  }
  return ret;
}

int ObExprVecIVFCenterID::calc_center_id_batch_cpu(const ObExpr &expr, ObEvalCtx &ctx,
                                                  const ObBitVector &skip, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  for (uint32_t a = 0; OB_SUCC(ret) && a < expr.arg_cnt_; a++) {
    if (OB_FAIL(expr.args_[a]->eval_batch(ctx, skip, batch_size))) {
      LOG_WARN("eval batch arg failed", K(ret), K(a));
      break;
    }
  }
  if (OB_SUCC(ret)) {
    ObDatum *results = expr.locate_batch_datums(ctx);
    ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
    if (OB_ISNULL(results)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("batch results is null", K(ret));
    } else {
      ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
      for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; i++) {
        if (skip.at(i) || eval_flags.at(i)) continue;
        batch_guard.set_batch_idx(i);
        if (OB_FAIL(calc_center_id(expr, ctx, results[i]))) {
          LOG_WARN("calc_center_id failed", K(ret), K(i));
        } else {
          eval_flags.set(i);
        }
      }
    }
  }
  return ret;
}

int ObExprVecIVFCenterID::calc_center_id_vector_cpu(const ObExpr &expr, ObEvalCtx &ctx,
                                                   const ObBitVector &skip, const EvalBound &bound)
{
  int ret = OB_SUCCESS;
  for (uint32_t a = 0; OB_SUCC(ret) && a < expr.arg_cnt_; a++) {
    if (OB_FAIL(expr.args_[a]->eval_vector(ctx, skip, bound))) {
      LOG_WARN("eval vector arg failed", K(ret), K(a));
      break;
    }
  }
  if (OB_SUCC(ret)) {
    ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
    ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
    batch_guard.set_batch_size(bound.batch_size());
    for (int64_t i = bound.start(); OB_SUCC(ret) && i < bound.end(); i++) {
      if (skip.at(i) || eval_flags.at(i)) continue;
      batch_guard.set_batch_idx(i);
      ObDatum &res = expr.locate_datum_for_write(ctx);
      if (OB_FAIL(calc_center_id(expr, ctx, res))) {
        LOG_WARN("calc_center_id failed", K(ret), K(i));
      } else {
        eval_flags.set(i);
      }
    }
  }
  return ret;
}

#ifdef __APPLE__
static int calc_center_id_batch_metal_impl(const ObExpr &expr, ObEvalCtx &ctx,
    const ObBitVector &skip, const int64_t batch_size,
    ObTableID table_id, ObTabletID tablet_id, ObVectorIndexDistAlgorithm dis_algo,
    const int64_t dim, const int64_t n_centers,
    const float *centers_buf, float *queries_buf, float *distances_buf)
{
  int ret = OB_SUCCESS;
  ObDatum *d0 = expr.args_[0]->locate_batch_datums(ctx);
  ObDatum *results = expr.locate_batch_datums(ctx);
  ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();

  int64_t compact = 0;
  for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; i++) {
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
    if (OB_ISNULL(row_arr) || row_arr->size() != static_cast<uint64_t>(dim)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("vector dim mismatch", K(ret), K(i));
      break;
    }
    memcpy(queries_buf + compact * dim, row_arr->get_data(), static_cast<size_t>(dim) * sizeof(float));
    compact++;
  }
  if (OB_FAIL(ret) || compact == 0) return ret;

  if (n_centers == 0) {
    for (int64_t i = 0; i < batch_size; i++) {
      if (skip.at(i) || eval_flags.at(i)) continue;
      if (d0[i].is_null()) continue;
      int64_t buf_len = OB_DOC_ID_COLUMN_BYTE_LENGTH;
      char *buf = expr.get_str_res_mem(ctx, buf_len);
      ObString str(buf_len, 0, buf);
      ObCenterId center_id(tablet_id.id(), 1);
      if (OB_FAIL(ObVectorClusterHelper::set_center_id_to_string(center_id, str))) {
        LOG_WARN("set_center_id_to_string failed", K(ret));
      } else {
        results[i].set_string(str);
        eval_flags.set(i);
      }
    }
    return ret;
  }

  int gpu_ret = gpu_acc::vector_metal::metal_l2_batch_query_vids(
      queries_buf, centers_buf, compact, n_centers, dim, distances_buf);
  if (gpu_ret != 0) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("metal_l2_batch_query_vids failed", K(gpu_ret));
    return ret;
  }

  compact = 0;
  for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; i++) {
    if (skip.at(i) || eval_flags.at(i)) continue;
    if (d0[i].is_null()) continue;
    const float *row_dist = distances_buf + compact * n_centers;
    int64_t best = 0;
    float best_d = row_dist[0];
    for (int64_t c = 1; c < n_centers; c++) {
      if (row_dist[c] < best_d) {
        best_d = row_dist[c];
        best = c;
      }
    }
    int64_t center_id_1based = best + 1;
    int64_t buf_len = OB_DOC_ID_COLUMN_BYTE_LENGTH;
    char *buf = expr.get_str_res_mem(ctx, buf_len);
    ObString str(buf_len, 0, buf);
    ObCenterId center_id(tablet_id.id(), center_id_1based);
    if (OB_FAIL(ObVectorClusterHelper::set_center_id_to_string(center_id, str))) {
      LOG_WARN("set_center_id_to_string failed", K(ret));
    } else {
      results[i].set_string(str);
      eval_flags.set(i);
    }
    compact++;
  }
  return ret;
}

// Metal path: regardless of batch size, build query and center matrices and run Metal;
// fall back to CPU only when Metal cannot be used (bad args, null, COS, alloc fail) or Metal returns error.
int ObExprVecIVFCenterID::calc_center_id_batch_metal(const ObExpr &expr, ObEvalCtx &ctx,
                                                    const ObBitVector &skip, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  bool fallback_to_cpu = false;

  if (expr.arg_cnt_ != 4) {
    fallback_to_cpu = true;
  } else {
    if (OB_FAIL(expr.args_[0]->eval_batch(ctx, skip, batch_size))) {
      LOG_WARN("eval batch arg0 failed", K(ret));
    } else if (OB_FAIL(expr.args_[1]->eval_batch(ctx, skip, batch_size))) {
      LOG_WARN("eval batch arg1 failed", K(ret));
    } else if (OB_FAIL(expr.args_[2]->eval_batch(ctx, skip, batch_size))) {
      LOG_WARN("eval batch arg2 failed", K(ret));
    } else if (OB_FAIL(expr.args_[3]->eval_batch(ctx, skip, batch_size))) {
      LOG_WARN("eval batch arg3 failed", K(ret));
    } else {
      ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
      int64_t valid_cnt = 0;
      for (int64_t i = 0; i < batch_size; i++) {
        if (!skip.at(i) && !eval_flags.at(i)) {
          valid_cnt++;
        }
      }
      if (valid_cnt <= 0) {
        fallback_to_cpu = true;
      } else {
        ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
        common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();
        ObTableID table_id = OB_INVALID_ID;
        ObTabletID tablet_id;
        ObVectorIndexDistAlgorithm dis_algo = VIDA_MAX;
        bool contain_null = false;
        ObIArrayType *arr = nullptr;
        ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
        batch_guard.set_batch_idx(0);

        float *centers_buf = nullptr;
        int64_t n_centers = 0;
        int64_t dim = 0;
        if (OB_FAIL(ObVectorIndexUtil::eval_ivf_centers_common_metal(
                alloc, expr, ctx, centers_buf, n_centers, dim,
                table_id, tablet_id, dis_algo, contain_null, arr))) {
          LOG_WARN("eval_ivf_centers_common_metal failed", K(ret));
          fallback_to_cpu = true;
        } else if (contain_null || OB_ISNULL(arr) || arr->size() == 0) {
          fallback_to_cpu = true;
        } else if (dis_algo == VIDA_COS) {
          fallback_to_cpu = true;
        } else if (n_centers <= 0 || OB_ISNULL(centers_buf)) {
          fallback_to_cpu = true;
        } else {
          const uint64_t qbuf_size = static_cast<uint64_t>(valid_cnt) * static_cast<uint64_t>(dim) * sizeof(float);
          const uint64_t dbuf_size = static_cast<uint64_t>(valid_cnt) * static_cast<uint64_t>(n_centers) * sizeof(float);
          // Arena allocation, CPU-aligned; Apple unified memory lets Metal access without extra copy.
          float *queries_buf = static_cast<float *>(alloc.alloc(qbuf_size));
          float *distances_buf = static_cast<float *>(alloc.alloc(dbuf_size));
          if (OB_ISNULL(queries_buf) || OB_ISNULL(distances_buf)) {
            fallback_to_cpu = true;
          } else {
            ret = calc_center_id_batch_metal_impl(expr, ctx, skip, batch_size,
                table_id, tablet_id, dis_algo, dim, n_centers, centers_buf, queries_buf, distances_buf);
            if (OB_FAIL(ret)) {
              fallback_to_cpu = true;
              ret = OB_SUCCESS;
            }
          }
        }
      }
    }
  }

  if (fallback_to_cpu) {
    ret = calc_center_id_batch_cpu(expr, ctx, skip, batch_size);
  }
  return ret;
}

// Metal path: regardless of batch size, build query and center matrices and run Metal;
// fall back to CPU only when Metal cannot be used or Metal returns error. Single return at end.
int ObExprVecIVFCenterID::calc_center_id_vector_metal(const ObExpr &expr, ObEvalCtx &ctx,
                                                     const ObBitVector &skip, const EvalBound &bound)
{
  int ret = OB_SUCCESS;
  bool fallback_to_cpu = false;

  if (expr.arg_cnt_ != 4) {
    fallback_to_cpu = true;
  } else {
    const int64_t batch_size = bound.batch_size();
    if (OB_FAIL(expr.args_[0]->eval_vector(ctx, skip, bound))) {
      LOG_WARN("eval vector arg0 failed", K(ret));
    } else if (OB_FAIL(expr.args_[1]->eval_vector(ctx, skip, bound))) {
      LOG_WARN("eval vector arg1 failed", K(ret));
    } else if (OB_FAIL(expr.args_[2]->eval_vector(ctx, skip, bound))) {
      LOG_WARN("eval vector arg2 failed", K(ret));
    } else if (OB_FAIL(expr.args_[3]->eval_vector(ctx, skip, bound))) {
      LOG_WARN("eval vector arg3 failed", K(ret));
    } else {
      ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
      int64_t valid_cnt = 0;
      for (int64_t i = bound.start(); i < bound.end(); i++) {
        if (!skip.at(i) && !eval_flags.at(i)) {
          valid_cnt++;
        }
      }
      common::ObIVector *vec0 = expr.args_[0]->get_vector(ctx);
      if (OB_ISNULL(vec0)) {
        fallback_to_cpu = true;
      } else if (valid_cnt <= 0) {
        fallback_to_cpu = true;
      } else {
        ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
        common::ObArenaAllocator &alloc = tmp_alloc_g.get_allocator();
        ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
        batch_guard.set_batch_size(batch_size);
        batch_guard.set_batch_idx(bound.start());
        ObTableID table_id = OB_INVALID_ID;
        ObTabletID tablet_id;
        ObVectorIndexDistAlgorithm dis_algo = VIDA_MAX;
        bool contain_null = false;
        ObIArrayType *arr = nullptr;

        float *centers_buf = nullptr;
        int64_t n_centers = 0;
        int64_t dim = 0;
        if (OB_FAIL(ObVectorIndexUtil::eval_ivf_centers_common_metal(
                alloc, expr, ctx, centers_buf, n_centers, dim,
                table_id, tablet_id, dis_algo, contain_null, arr))) {
          fallback_to_cpu = true;
        } else if (contain_null || OB_ISNULL(arr) || arr->size() == 0 || dis_algo == VIDA_COS) {
          fallback_to_cpu = true;
        } else {
          if (n_centers <= 0 || dim <= 0 || OB_ISNULL(centers_buf)) {
            fallback_to_cpu = true;
          } else {
            uint64_t qbuf_size = static_cast<uint64_t>(valid_cnt) * static_cast<uint64_t>(dim) * sizeof(float);
            uint64_t dbuf_size = static_cast<uint64_t>(valid_cnt) * static_cast<uint64_t>(n_centers) * sizeof(float);
            // Use Arena to allocate, align with CPU, use Apple unified memory to access Metal directly, avoid extra copy.
            float *queries_buf = static_cast<float *>(alloc.alloc(qbuf_size));
            float *distances_buf = static_cast<float *>(alloc.alloc(dbuf_size));
            if (OB_ISNULL(queries_buf) || OB_ISNULL(distances_buf)) {
              fallback_to_cpu = true;
            } else {
              ObIArrayType *row_arr = nullptr;
              int64_t compact = 0;
              for (int64_t i = bound.start(); OB_SUCC(ret) && i < bound.end(); i++) {
                if (skip.at(i) || eval_flags.at(i) || vec0->is_null(i)) {
                  continue;
                }
                ObString s = vec0->get_string(i);
                ObDatum d;
                d.set_string(s);
                if (OB_FAIL(ObArrayExprUtils::get_type_vector(*(expr.args_[0]), d, ctx, alloc, row_arr))) {
                  LOG_WARN("get_type_vector failed", K(ret), K(i));
                  break;
                }
                if (OB_ISNULL(row_arr) || row_arr->size() != static_cast<uint64_t>(dim)) {
                  ret = OB_ERR_UNEXPECTED;
                  break;
                }
                memcpy(queries_buf + compact * dim, row_arr->get_data(), static_cast<size_t>(dim) * sizeof(float));
                compact++;
              }
              if (OB_FAIL(ret)) {
                fallback_to_cpu = true;
                ret = OB_SUCCESS;
              } else if (compact == 0) {
                fallback_to_cpu = true;
              } else {
                int gpu_ret = gpu_acc::vector_metal::metal_l2_batch_query_vids(
                    queries_buf, centers_buf, compact, n_centers, dim, distances_buf);
                if (gpu_ret != 0) {
                  fallback_to_cpu = true;
                } else {
                  ObDatum *results = expr.locate_batch_datums(ctx);
                  if (OB_ISNULL(results)) {
                    fallback_to_cpu = true;
                  } else {
                    compact = 0;
                    for (int64_t i = bound.start(); OB_SUCC(ret) && i < bound.end(); i++) {
                      if (skip.at(i) || eval_flags.at(i)) continue;
                      if (vec0->is_null(i)) {
                        results[i].set_null();
                        eval_flags.set(i);
                        continue;
                      }
                      const float *row_dist = distances_buf + compact * n_centers;
                      int64_t best = 0;
                      float best_d = row_dist[0];
                      for (int64_t c = 1; c < n_centers; c++) {
                        if (row_dist[c] < best_d) {
                          best_d = row_dist[c];
                          best = c;
                        }
                      }
                      int64_t center_id_1based = best + 1;
                      int64_t buf_len = OB_DOC_ID_COLUMN_BYTE_LENGTH;
                      char *buf = expr.get_str_res_mem(ctx, buf_len);
                      ObString str(buf_len, 0, buf);
                      ObCenterId center_id(tablet_id.id(), center_id_1based);
                      if (OB_FAIL(ObVectorClusterHelper::set_center_id_to_string(center_id, str))) {
                        LOG_WARN("set_center_id_to_string failed", K(ret));
                      } else {
                        results[i].set_string(str);
                        eval_flags.set(i);
                      }
                      compact++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (fallback_to_cpu) {
    ret = calc_center_id_vector_cpu(expr, ctx, skip, bound);
  }
  return ret;
}
#endif

}  // namespace sql
}  // namespace oceanbase

