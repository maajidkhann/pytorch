import torch

def test_aspt_integration():
    # Create a sample dense matrix and input vector
    rows, cols = 128, 128
    dense = torch.randn(rows, cols, dtype=torch.float32)
    vin = torch.randn(cols, 4, dtype=torch.float32)  # for SpMM

    # Run ASpT inspection
    try:
        meta = torch.ops.aten.aspt_inspect(dense)
        print("✅ aspt_inspect ran successfully.")
    except Exception as e:
        print("❌ aspt_inspect failed:", e)
        return

    # Unpack metadata
    if len(meta) != 9:
        print("❌ Expected 9 tensors from aspt_inspect, got", len(meta))
        return

    try:
        # Run ASpT execution (SpMM)
        vout = torch.ops.aten.aspt_execute(*meta, vin)
        print("✅ aspt_execute ran successfully.")
        print("Output shape:", vout.shape)
        #print("Output:", vout)
    except Exception as e:
        print("❌ aspt_execute failed:", e)

if __name__ == "__main__":
    test_aspt_integration()

