import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import time
import copy
import gc


class ASpTLinear(nn.Module):
    def __init__(self, dense_weight, dense_bias=None, name=""):
        super().__init__()
        self.meta = torch.ops.aten.aspt_inspect(dense_weight.contiguous())
        self.bias = dense_bias
        self.name = name
        self.layer_time_ms = 0.0

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x_t = x.transpose(0, 1).contiguous()
        t0 = time.perf_counter()
        out_t = torch.ops.aten.aspt_execute(*self.meta, x_t)
        t1 = time.perf_counter()
        self.layer_time_ms += (t1 - t0) * 1000
        out = out_t.transpose(0, 1)
        if self.bias is not None:
            out += self.bias
        if len(original_shape) == 3:
            out = out.reshape(original_shape[0], original_shape[1], -1)
        return out


def apply_wanda_pruning(model, sparsity=0.9):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                W = module.weight.data
                row_norms = torch.norm(W, p=2, dim=1)
                k = int(W.size(0) * (1 - sparsity))
                if k < 1:
                    continue
                topk = torch.topk(row_norms, k, sorted=False).indices
                mask = torch.zeros_like(row_norms)
                mask[topk] = 1.0
                mask = mask.unsqueeze(1).expand_as(W)
                module.weight.data *= mask


def replace_linear_with_aspt(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                ASpTLinear(module.weight.data, module.bias.data if module.bias is not None else None, name)
            )
        else:
            replace_linear_with_aspt(module)


def benchmark_dense_vs_aspt_cpu():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    base_model = BertModel.from_pretrained("bert-base-uncased").eval()
    inputs = tokenizer("ASpT vs dense: benchmarking with WANDA pruning", return_tensors="pt")

    for sparsity in [0.5, 0.7, 0.9]:
        print(f"\n🔧 Sparsity Level: {int(sparsity * 100)}%")

        dense_model = copy.deepcopy(base_model)
        aspt_model = copy.deepcopy(base_model)

        #apply_wanda_pruning(dense_model, sparsity)
        apply_wanda_pruning(aspt_model, sparsity)
        replace_linear_with_aspt(aspt_model)

        with torch.no_grad():
            for _ in range(3):
                dense_model(**inputs)
                aspt_model(**inputs)

        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(10):
                out_dense = dense_model(**inputs)
            t1 = time.perf_counter()

        with torch.no_grad():
            t2 = time.perf_counter()
            for _ in range(10):
                out_aspt = aspt_model(**inputs)
            t3 = time.perf_counter()

        dense_time = (t1 - t0) / 10 * 1000
        aspt_time = (t3 - t2) / 10 * 1000
        speedup = dense_time / aspt_time

        diff = (out_dense.last_hidden_state - out_aspt.last_hidden_state).abs()
        max_error = diff.max().item()
        mean_error = diff.mean().item()

        print(f"⏱ Dense Time : {dense_time:.3f} ms")
        print(f"⚡ ASpT  Time : {aspt_time:.3f} ms")
        print(f"🚀 Speedup    : {speedup:.2f}x")
        print(f"🎯 Max Error  : {max_error:.4e}")
        print(f"🎯 Mean Error : {mean_error:.4e}")
        
        '''
        print("📊 ASpT Per-layer timing:")
        for name, module in aspt_model.named_modules():
            if isinstance(module, ASpTLinear):
                print(f"  - {module.name:30s}: {module.layer_time_ms:.3f} ms")
        '''

        del dense_model, aspt_model, out_dense, out_aspt
        gc.collect()


if __name__ == "__main__":
    benchmark_dense_vs_aspt_cpu()
