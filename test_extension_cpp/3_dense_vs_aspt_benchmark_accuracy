import torch

def test_aspt_integration():
    rows, cols = 256, 256
    batch_size = 8
    dense = torch.randn(rows, cols, dtype=torch.float32)
    vin = torch.randn(batch_size, cols, dtype=torch.float32)  # [8, 256]

    print("Input shape (vin):", vin.shape)

    try:
        meta = torch.ops.aten.aspt_inspect(dense)
        print("✅ aspt_inspect ran successfully.")
    except Exception as e:
        print("❌ aspt_inspect failed:", e)
        return

    if len(meta) != 9:
        print("❌ Expected 9 tensors from aspt_inspect, got", len(meta))
        return

    try:
        vout_aspt = torch.ops.aten.aspt_execute(*meta, vin).transpose(0, 1)  # [8, 256]
        print("✅ aspt_execute ran successfully.")
        print("Output shape (ASpT):", vout_aspt.shape)
    except Exception as e:
        print("❌ aspt_execute failed:", e)
        return

    try:
        vout_dense = vin @ dense.T  # [8, 256]
        print("✅ Dense matmul ran successfully.")
        print("Output shape (Dense):", vout_dense.shape)
    except Exception as e:
        print("❌ Dense matmul failed:", e)
        return

    # Compare outputs
    l2_diff = torch.norm(vout_aspt - vout_dense).item()
    max_diff = (vout_aspt - vout_dense).abs().max().item()
    same = torch.allclose(vout_aspt, vout_dense, atol=1e-4, rtol=1e-4)

    print("\n🔍 Comparison:")
    print(f"L2 difference:      {l2_diff:.6f}")
    print(f"Max absolute diff:  {max_diff:.6f}")
    print(f"Match (allclose)?   {'✅ Yes' if same else '❌ No'}")

if __name__ == "__main__":
    test_aspt_integration()
