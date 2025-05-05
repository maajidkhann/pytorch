import torch
from torch.utils import benchmark

def benchmark_aspt(shapes, sparsity_levels, dtype=torch.float32):
    for M, K, N in shapes:
        print(f"\n🧩 Shape: ({M}×{K}) × ({K}×{N}) → ({M}×{N})")
        for sparsity in sparsity_levels:
            torch.manual_seed(0)

            # vin is the M×K matrix to be sparsified
            vin = torch.randn(M, K, dtype=dtype)
            mask = torch.rand_like(vin) > sparsity
            vin *= mask

            # dense second operand
            dense = torch.randn(K, N, dtype=dtype)

            # reference
            vref = vin @ dense

            # time the plain dense matmul
            t_dense = benchmark.Timer(
                stmt='vin @ dense',
                globals={'vin': vin, 'dense': dense}
            ).timeit(50)

            # prepare sparse representation of vin
            meta = torch.ops.aten.aspt_inspect(vin)

            # time the ASpT matmul
            def run_aspt():
                return torch.ops.aten.aspt_execute(*meta, dense)

            t_aspt = benchmark.Timer(
                stmt='run_aspt()',
                globals={'run_aspt': run_aspt}
            ).timeit(50)

            # correctness check
            vout = run_aspt()
            correct = torch.allclose(vref, vout, rtol=1e-3, atol=1e-4)
            diff = (vref - vout).abs()
            max_error = diff.max().item()
            mean_error = diff.mean().item()

            # report
            dense_ms = t_dense.mean * 1e3
            aspt_ms = t_aspt.mean * 1e3
            speedup = dense_ms / aspt_ms if aspt_ms > 0 else float('inf')

            print(f"\n🔹 Sparsity: {sparsity:.1f}")
            print(f"✅ Accuracy: {correct}")
            print(f"🔍 Max Error: {max_error:.2e} | Mean Error: {mean_error:.2e}")
            print(f"⏱ Dense: {dense_ms:.3f} ms | ASpT: {aspt_ms:.3f} ms")
            print(f"🚀 Speedup: {speedup:.2f}×")

if __name__ == "__main__":
    shapes = [
    #     (11008, 4096, 128),
    #     (4096, 11008, 128),
        (4096, 4096, 128),
    ]
    sparsity_levels = [0.6, 0.7, 0.8, 0.9]
    benchmark_aspt(shapes, sparsity_levels)
