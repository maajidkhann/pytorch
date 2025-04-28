#include <ATen/ATen.h>
#include <torch/library.h>

namespace extension_cpp {

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

// Add two tensors into an output tensor (in-place)
void myadd_out_cpu(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
  TORCH_CHECK(b.sizes() == out.sizes(), "Output tensor size must match inputs");
  TORCH_CHECK(a.dtype() == at::kFloat, "Input tensor 'a' must be float");
  TORCH_CHECK(b.dtype() == at::kFloat, "Input tensor 'b' must be float");
  TORCH_CHECK(out.dtype() == at::kFloat, "Output tensor 'out' must be float");
  TORCH_CHECK(out.is_contiguous(), "Output tensor 'out' must be contiguous");
  TORCH_INTERNAL_ASSERT(a.device().is_cpu(), "Tensor 'a' must be on CPU");
  TORCH_INTERNAL_ASSERT(b.device().is_cpu(), "Tensor 'b' must be on CPU");
  TORCH_INTERNAL_ASSERT(out.device().is_cpu(), "Tensor 'out' must be on CPU");

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();

  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();

  for (int64_t i = 0; i < out.numel(); i++) {
    result_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

// Register operators with dispatcher
TORCH_LIBRARY(extension_cpp, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);
}

} // namespace extension_cpp

