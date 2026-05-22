# Run this to check your average output token length
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-base")
df = pd.read_csv("data/cleaned/val_clean.csv")

lengths = df["output"].astype(str).apply(
    lambda x: len(tokenizer.encode(x, truncation=True, max_length=192))
)
print(f"Average output tokens: {lengths.mean():.1f}")
print(f"Median output tokens:  {lengths.median():.1f}")
print(f"Max output tokens:     {lengths.max()}")
print(f"\nExpected true loss = reported_loss ÷ {lengths.mean():.1f}")
print(f"Your current loss  = 57 ÷ {lengths.mean():.1f} = {57/lengths.mean():.2f} per token")