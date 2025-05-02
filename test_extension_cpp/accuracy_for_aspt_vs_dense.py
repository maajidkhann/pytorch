import torch

def test_aspt_accuracy():
    torch.manual_seed(42)
    rows, cols, batch = 128, 128, 4

    dense = torch.randn(rows, cols, dtype=torch.float32)
    dense[dense.abs() < 0.2] = 0  # sparsify

    vin = torch.randn(cols, batch, dtype=torch.float32)

    # Reference dense matmul
    vref = dense @ vin

    # ASpT path
    meta = torch.ops.aten.aspt_inspect(dense)
    vout = torch.ops.aten.aspt_execute(*meta, vin)

    print("Dense vs ASpT:", torch.allclose(vref, vout, rtol=1e-3, atol=1e-4))
    print("Max Abs Diff:", (vref - vout).abs().max().item())

test_aspt_accuracy()

