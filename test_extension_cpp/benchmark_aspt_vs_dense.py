import torch
import torch.utils.benchmark as benchmark

def benchmark_aspt(batch=4, dtype=torch.float32):
    matrix_sizes = [512, 1024, 2048]
    sparsity_levels = [0.6, 0.7, 0.8, 0.9]

    for size in matrix_sizes:
        print(f"\n🧩 Matrix Size: {size}x{size}")
        for sparsity in sparsity_levels:
            torch.manual_seed(0)

            dense = torch.randn(size, size, dtype=dtype)
            mask = torch.rand_like(dense) > sparsity
            dense *= mask

            vin = torch.randn(size, batch, dtype=dtype)
            vref = dense @ vin

            t_dense = benchmark.Timer(
                stmt='dense @ vin',
                globals={'dense': dense, 'vin': vin}
            ).timeit(50)

            meta = torch.ops.aten.aspt_inspect(dense)

            def run_aspt():
                return torch.ops.aten.aspt_execute(*meta, vin)

            t_aspt = benchmark.Timer(
                stmt='run_aspt()',
                globals={'run_aspt': run_aspt}
            ).timeit(50)

            vout = torch.ops.aten.aspt_execute(*meta, vin)
            correct = torch.allclose(vref, vout, rtol=1e-3, atol=1e-4)
            diff = (vref - vout).abs()
            max_error = diff.max().item()
            mean_error = diff.mean().item()

            dense_time = t_dense.mean * 1e3
            aspt_time = t_aspt.mean * 1e3
            speedup = dense_time / aspt_time if aspt_time > 0 else float('inf')

            print(f"\n🔹 Sparsity: {sparsity:.1f}")
            print(f"✅ Accuracy: {correct}")
            print(f"🔍 Max Error: {max_error:.2e} | Mean Error: {mean_error:.2e}")
            print(f"⏱ Dense: {dense_time:.3f} ms | ASpT: {aspt_time:.3f} ms")
            print(f"🚀 Speedup: {speedup:.2f}x")

# Run the benchmark
benchmark_aspt()
