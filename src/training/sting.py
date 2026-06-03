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
print(f"Your current loss  = 57 ÷ {lengths.mean():.1f} = {57/lengths.mean():.2f} per token)

Length: 196 chars, 24 words
Full: Trichomoniasis is a sexually transmitted infection caused by the trichomonas vaginalis bacteria. It can be cured with antibiotics, including metronidazole (Tylenol) or azithromycin (Azithromycin).
Ends with: ...'Tylenol) or azithromycin (Azithromycin).'
------------------------------------------------------------
Length: 235 chars, 40 words
Full: This is a question about, Sexual Health. It's important to practice safe sex when having sex with someone who is not in a relationship with you. Here are some steps you can take to protect yourself from sexually transmitted infections.
Ends with: ...'lf from sexually transmitted infections.'
------------------------------------------------------------
Length: 185 chars, 33 words
Full: This is a question about, Herpes. Hepatitis B is transmitted when blood, semen, or other body fluids from a person infected with the virus enter the body of someone who is not infected.
Ends with: ...'the body of someone who is not infected.'
------------------------------------------------------------