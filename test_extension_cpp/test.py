import torch

a = torch.ones(5, dtype=torch.float32)
b = torch.ones(5, dtype=torch.float32)

c = torch.ops.extension_cpp.mymuladd(a, b, 2.0)
d = torch.ops.extension_cpp.mymul(a, b)

print("mymuladd result:", c)
print("mymul result:", d)

# Auto verify
assert torch.allclose(c, torch.full_like(a, 3.0))
assert torch.allclose(d, torch.ones_like(a))

print("All tests passed successfully!")
