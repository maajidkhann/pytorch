#pragma once

namespace at {
namespace native {

// Metadata structure for ASpT layout
template<typename Scalar>
struct InspectorMetadata {
    int npanel;
    int nr;

    int* mcsr_e;
    int* mcsr_cnt;
    int* mcsr_chk;

    int* row_ptrs_padded;
    int* col_indices_reordered;
    Scalar* values_reordered;

    double avg;
    double vari;
    int nThread;

    int* special;
    int* special2;
    int special_p;
};

// Converts a dense matrix to ASpT metadata
InspectorMetadata<float> inspect(float* a, int nr0, int nc);

// Performs SpMM using ASpT metadata and dense input
void execute(
    const InspectorMetadata<float>& meta,
    int nr0, int nc, int sc,
    float* vin,
    float* vout
);

// Wrapper for PyTorch integration
std::vector<at::Tensor> aspt_inspect(const at::Tensor& dense);

at::Tensor aspt_execute(
    const at::Tensor& mcsr_e,
    const at::Tensor& mcsr_cnt,
    const at::Tensor& mcsr_chk,
    const at::Tensor& row_ptrs,
    const at::Tensor& col_idx,
    const at::Tensor& values,
    const at::Tensor& avg_tensor,
    const at::Tensor& vari_tensor,
    const at::Tensor& nThread_tensor,
    const at::Tensor& vin
);

} // namespace native
} // namespace at
