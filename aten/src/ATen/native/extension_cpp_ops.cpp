#include <ATen/ATen.h>
#include <torch/library.h>

namespace extension_cpp {
namespace native {

// Multiply two tensors and add a scalar
at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
  TORCH_CHECK(a.dtype() == at::kFloat, "Input tensor 'a' must be float");
  TORCH_CHECK(b.dtype() == at::kFloat, "Input tensor 'b' must be float");
  TORCH_INTERNAL_ASSERT(a.device().is_cpu(), "Tensor 'a' must be on CPU");
  TORCH_INTERNAL_ASSERT(b.device().is_cpu(), "Tensor 'b' must be on CPU");

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty_like(a_contig);

  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + static_cast<float>(c);
  }
  return result;
}

// Multiply two tensors element-wise
at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
  TORCH_CHECK(a.dtype() == at::kFloat, "Input tensor 'a' must be float");
  TORCH_CHECK(b.dtype() == at::kFloat, "Input tensor 'b' must be float");
  TORCH_INTERNAL_ASSERT(a.device().is_cpu(), "Tensor 'a' must be on CPU");
  TORCH_INTERNAL_ASSERT(b.device().is_cpu(), "Tensor 'b' must be on CPU");

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty_like(a_contig);

  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}

} // namespace native
} // namespace extension_cpp
