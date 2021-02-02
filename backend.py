import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "custom-bert-base-uncased", num_labels=2
)
model.eval()


@torch.no_grad()
def predict(text):
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    output = model(**tokens)
    logits = output["logits"].detach()
    prediction = logits.softmax(dim=1)[0][1].item()
    return prediction

