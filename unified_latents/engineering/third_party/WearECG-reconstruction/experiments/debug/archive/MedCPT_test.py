import torch
from transformers import AutoTokenizer, AutoModel
import os

os.environ["HF_HUB_OFFLINE"] = "1"

model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

queries = [
    "diabetes treatment",
    "How to treat diabetes?",
    "A 45-year-old man presents with increased thirst and frequent urination over the past 3 months.",
    "NMSL"
]

with torch.no_grad():
    # tokenize the queries
    encoded = tokenizer(
        queries,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
    )

    # encode the queries (use the [CLS] last hidden states as the representations)
    embeds = model(**encoded).last_hidden_state[:, 0, :]

    print(embeds)
    print(embeds.size())
