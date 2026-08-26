import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full list of labeled images
df = pd.read_csv("data/train.csv")

print(f"Total images: {len(df)}")
print("\nClass distribution (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative):")
print(df["diagnosis"].value_counts().sort_index())

# Split into 80% train, 20% validation
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["diagnosis"]
)

print(f"\nTrain set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")

train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)

print("\nSaved: data/train_split.csv and data/val_split.csv")