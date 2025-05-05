#include <ATen/ATen.h>
#include <vector>
#include <omp.h>
#include "AsPT.h"

namespace at {
namespace native {

InspectorMetadata<float> inspect(float* a, int nr0, int nc) {
    constexpr int BLOCK_HEIGHT = 128;
    InspectorMetadata<float> meta;

    meta.npanel = (nr0 + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    meta.nr = nr0;
    meta.nThread = 1;

    std::vector<int> row_ptr(nr0 + 1, 0);
    std::vector<int> col_idx_tmp;
    std::vector<float> values_tmp;

    // Build CSR from dense
    for (int i = 0; i < nr0; ++i) {
        for (int j = 0; j < nc; ++j) {
            float val = a[i * nc + j];
            if (val != 0.0f) {
                col_idx_tmp.push_back(j);
                values_tmp.push_back(val);
                row_ptr[i + 1]++;
            }
        }
        row_ptr[i + 1] += row_ptr[i];
    }

    int nnz = values_tmp.size();

    // Allocate & copy
    meta.row_ptrs_padded = new int[nr0 + 1];
    meta.col_indices_reordered = new int[nnz];
    meta.values_reordered = new float[nnz];

    std::copy(row_ptr.begin(), row_ptr.end(), meta.row_ptrs_padded);
    std::copy(col_idx_tmp.begin(), col_idx_tmp.end(), meta.col_indices_reordered);
    std::copy(values_tmp.begin(), values_tmp.end(), meta.values_reordered);

    // Dummy unused metadata
    meta.mcsr_e = new int[meta.npanel * BLOCK_HEIGHT * 8]{};
    meta.mcsr_cnt = new int[meta.npanel + 1]{};
    meta.mcsr_chk = new int[meta.npanel + 1]{};
    meta.avg = 0.5;
    meta.vari = 0.1;

    return meta;
}

void execute(const InspectorMetadata<float>& meta, int nr0, int nc, int sc, float* vin, float* vout) {
    #pragma omp parallel for
    for (int row = 0; row < nr0; ++row) {
        for (int j = meta.row_ptrs_padded[row]; j < meta.row_ptrs_padded[row + 1]; ++j) {
            int col = meta.col_indices_reordered[j];
            float val = meta.values_reordered[j];
            for (int k = 0; k < sc; ++k) {
                vout[row * sc + k] += val * vin[col * sc + k];
            }
        }
    }
    //std::cout << "@@@@@@Entered Execute!!!" << std::endl;
}

std::vector<Tensor> aspt_inspect(const Tensor& dense) {
    TORCH_CHECK(dense.dim() == 2, "Expected 2D tensor");
    auto a_contig = dense.contiguous();
    float* a_ptr = a_contig.data_ptr<float>();

    int rows = dense.size(0);
    int cols = dense.size(1);

    InspectorMetadata<float> meta = inspect(a_ptr, rows, cols);

    auto opts_i32 = at::device(kCPU).dtype(kInt);
    auto opts_f32 = at::device(kCPU).dtype(kFloat);

    int nnz = meta.row_ptrs_padded[rows];  // last value of row_ptrs gives total nnz

    Tensor mcsr_e   = from_blob(meta.mcsr_e, {meta.npanel * 128 * 8}, opts_i32).clone();
    Tensor mcsr_cnt = from_blob(meta.mcsr_cnt, {meta.npanel + 1}, opts_i32).clone();
    Tensor mcsr_chk = from_blob(meta.mcsr_chk, {meta.npanel + 1}, opts_i32).clone();
    Tensor row_ptrs = from_blob(meta.row_ptrs_padded, {meta.nr + 1}, opts_i32).clone();
    Tensor col_idx  = from_blob(meta.col_indices_reordered, {nnz}, opts_i32).clone();
    Tensor values   = from_blob(meta.values_reordered, {nnz}, opts_f32).clone();

    Tensor avg_tensor     = at::tensor({meta.avg}, opts_f32);
    Tensor vari_tensor    = at::tensor({meta.vari}, opts_f32);
    Tensor nThread_tensor = at::tensor({meta.nThread}, opts_i32);
    //std::cout << "@@@@@@Entered aspt_inspect!!!" << std::endl;

    return {mcsr_e, mcsr_cnt, mcsr_chk, row_ptrs, col_idx, values, avg_tensor, vari_tensor, nThread_tensor};
}

Tensor aspt_execute(
    const Tensor& mcsr_e,
    const Tensor& mcsr_cnt,
    const Tensor& mcsr_chk,
    const Tensor& row_ptrs,
    const Tensor& col_idx,
    const Tensor& values,
    const Tensor& avg_tensor,
    const Tensor& vari_tensor,
    const Tensor& nThread_tensor,
    const Tensor& vin
) {
    int nr = row_ptrs.size(0) - 1;
    int sc = vin.size(1);

    Tensor vout = at::zeros({nr, sc}, vin.options());

    InspectorMetadata<float> meta;
    meta.mcsr_e = mcsr_e.data_ptr<int>();
    meta.mcsr_cnt = mcsr_cnt.data_ptr<int>();
    meta.mcsr_chk = mcsr_chk.data_ptr<int>();
    meta.row_ptrs_padded = row_ptrs.data_ptr<int>();
    meta.col_indices_reordered = col_idx.data_ptr<int>();
    meta.values_reordered = values.data_ptr<float>();
    meta.avg = avg_tensor.item<float>();
    meta.vari = vari_tensor.item<float>();
    meta.nThread = nThread_tensor.item<int>();
    meta.npanel = mcsr_cnt.size(0) - 1;
    meta.nr = nr;

    execute(meta, nr, vin.size(0), sc, vin.data_ptr<float>(), vout.data_ptr<float>());
    //std::cout << "@@@@@@Entered aspt_execute!!!" << std::endl;
    return vout;
}

} // namespace native
} // namespace at
