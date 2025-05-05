import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import BertModel, BertTokenizer
import time


# === ASpTLinear: handles 2D/3D input, adds bias ===
class ASpTLinear(nn.Module):
    def __init__(self, dense_weight, dense_bias=None):
        super().__init__()
        self.meta = torch.ops.aten.aspt_inspect(dense_weight)
        self.bias = dense_bias

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, in_feat = x.shape
            x = x.reshape(-1, in_feat)

        x_t = x.transpose(0, 1).contiguous()
        out_t = torch.ops.aten.aspt_execute(*self.meta, x_t)
        out = out_t.transpose(0, 1)

        if self.bias is not None:
            out += self.bias

        if len(original_shape) == 3:
            out = out.reshape(original_shape[0], original_shape[1], -1)

        return out


# === Apply L1 pruning ===
def apply_l1_pruning_to_model(model, amount=0.9):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')


# === Replace nn.Linear with ASpTLinear ===
def replace_linear_with_aspt(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                ASpTLinear(module.weight.data, module.bias.data if module.bias is not None else None),
            )
        else:
            replace_linear_with_aspt(module)


# === Run and compare ===
def run_bert_comparison():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dense_model = BertModel.from_pretrained("bert-base-uncased").eval()
    aspt_model = BertModel.from_pretrained("bert-base-uncased").eval()

    # Prune and patch ASpT model
    apply_l1_pruning_to_model(aspt_model, amount=0.5)
    replace_linear_with_aspt(aspt_model)

    inputs = tokenizer("Hello from the ASpT-integrated BERT!", return_tensors="pt")

    # Run dense baseline
    with torch.no_grad():
        for _ in range(5):  # warmup
            dense_out = dense_model(**inputs)
        t0 = time.perf_counter()
        for _ in range(10):
            dense_out = dense_model(**inputs)
        t1 = time.perf_counter()

    # Run ASpT model
    with torch.no_grad():
        for _ in range(5):  # warmup
            aspt_out = aspt_model(**inputs)
        t2 = time.perf_counter()
        for _ in range(10):
            aspt_out = aspt_model(**inputs)
        t3 = time.perf_counter()

    # Timing
    dense_time = (t1 - t0) / 10 * 1000
    aspt_time = (t3 - t2) / 10 * 1000
    speedup = dense_time / aspt_time

    # Accuracy
    dense_tensor = dense_out.last_hidden_state
    aspt_tensor = aspt_out.last_hidden_state
    diff = (dense_tensor - aspt_tensor).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    dense_mean = dense_tensor.abs().mean().item()
    rel_error = mean_err / dense_mean if dense_mean > 0 else 0.0
    est_accuracy = 100.0 * (1.0 - rel_error)

    within_1e2 = (diff < 1e-2).float().mean().item() * 100
    within_1e3 = (diff < 1e-3).float().mean().item() * 100

    # Print summary
    print("\n🔍 BERT Inference Comparison:")
    print(f"⏱ Dense Time: {dense_time:.3f} ms")
    print(f"⚡ ASpT  Time: {aspt_time:.3f} ms")
    print(f"🚀 Speedup: {speedup:.2f}x")
    print(f"🎯 Max Abs Error: {max_err:.4e}")
    print(f"🎯 Mean Abs Error: {mean_err:.4e}")
    print(f"📊 Estimated Accuracy (~rel error): {est_accuracy:.2f}%")
    print(f"✅ % Values within 1e-2: {within_1e2:.2f}%")
    print(f"✅ % Values within 1e-3: {within_1e3:.2f}%")


if __name__ == "__main__":
    run_bert_comparison()
