import pandas as pd

# Load both datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add a 'label' column
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Concatenate and shuffle
df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# Save as combined file
df.to_csv('fake_or_real_news.csv', index=False)

print("✅ Combined dataset saved as fake_or_real_news.csv")
