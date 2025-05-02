import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import BertModel, BertTokenizer


# === ASpTLinear: handles 2D/3D input, adds bias ===
class ASpTLinear(nn.Module):
    def __init__(self, dense_weight, dense_bias=None):
        super().__init__()
        self.meta = torch.ops.aten.aspt_inspect(dense_weight)
        self.bias = dense_bias

    def forward(self, x):
        original_shape = x.shape

        # Handle 3D input [batch, seq_len, in_features]
        if x.dim() == 3:
            batch, seq_len, in_feat = x.shape
            x = x.reshape(-1, in_feat)  # [B*S, in_feat]

        # Transpose for ASpT
        x_t = x.transpose(0, 1)  # [in_features, B*S]
        out_t = torch.ops.aten.aspt_execute(*self.meta, x_t)  # [out_features, B*S]
        out = out_t.transpose(0, 1)  # [B*S, out_features]

        if self.bias is not None:
            out += self.bias

        if len(original_shape) == 3:
            out = out.reshape(original_shape[0], original_shape[1], -1)  # [B, S, out_features]

        return out


# === Prune all nn.Linear layers in model ===
def apply_l1_pruning_to_model(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # finalize mask


# === Replace all Linear layers recursively ===
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


# === Run inference ===
def run_bert_aspt_inference():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    # Apply 50% L1 pruning and ASpT substitution
    apply_l1_pruning_to_model(model, amount=0.5)
    replace_linear_with_aspt(model)

    # Sample input
    inputs = tokenizer("Hello from the ASpT-integrated BERT!", return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    print("\n✅ ASpT BERT inference completed.")
    print("Last hidden state shape:", outputs.last_hidden_state.shape)


if __name__ == "__main__":
    run_bert_aspt_inference()
