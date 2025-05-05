import torch
from transformers import BertTokenizer, BertModel

# Load pretrained tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Set to inference mode

# Prepare sample input
text = "Hello, this is a BERT inference test."
inputs = tokenizer(text, return_tensors="pt")

# Run 1 forward pass (inference iteration)
with torch.no_grad():
    outputs = model(**inputs)

# Print last hidden state (output of BERT)
print("Last hidden state shape:", outputs.last_hidden_state.shape)

