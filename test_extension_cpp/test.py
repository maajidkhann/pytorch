import torch
a = torch.ones(5, dtype=torch.float32)
b = torch.ones(5, dtype=torch.float32)
out = torch.empty_like(a)

print(torch.ops.extension_cpp.mymuladd(a, b, 2.0))  # (1 * 1) + 2 = 3
print(torch.ops.extension_cpp.mymul(a, b))          # 1 * 1 = 1
torch.ops.extension_cpp.myadd_out(a, b, out)
print(out)                                          # 1 + 1 = 2

