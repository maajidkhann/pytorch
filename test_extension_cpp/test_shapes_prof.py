import torch
from torch.utils import benchmark
from torch.profiler import profile, record_function, ProfilerActivity

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
            with profile(
                with_stack=True,
                profile_memory=True,
                record_shapes=False,
            ) as prof:
                vref = vin @ dense

            print(prof.key_averages(group_by_input_shape=False).table(row_limit=-1))

            # prepare sparse representation of vin
            meta = torch.ops.aten.aspt_inspect(vin)

            # time the ASpT matmul
            def run_aspt():
                return torch.ops.aten.aspt_execute(*meta, dense)

            with profile(
                with_stack=True,
                profile_memory=True,
                record_shapes=False,
            ) as prof:
                vout_aspt = run_aspt()
            print(prof.key_averages(group_by_input_shape=False).table(row_limit=-1))

            correct = torch.allclose(vref, vout_aspt, rtol=1e-2, atol=1e-3)   #looser tolerance (stricter tolerance is rtol=1e-4, atol=1e-4)
            diff = (vref - vout_aspt).abs()
            max_error = diff.max().item()
            mean_error = diff.mean().item()

            print(f"\n🔹 Sparsity: {sparsity:.1f}")
            print(f"✅ Accuracy: {correct}")
            print(f"🔍 Max Error: {max_error:.2e} | Mean Error: {mean_error:.2e}")

if __name__ == "__main__":
    shapes = [
         (11008, 4096, 128),
         (4096, 11008, 128),
        (4096, 4096, 128),
    ]
    sparsity_levels = [0.6, 0.7, 0.8, 0.9]
    benchmark_aspt(shapes, sparsity_levels)