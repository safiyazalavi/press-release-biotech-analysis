# imports
import pandas as pd, json
from nltk.tokenize import sent_tokenize

# json file path
input  = "cleaned_catalysts_cleaned_metadata.csv"
output = "data/original_data/biotech.json"

# read in csv 
df = pd.read_csv(input)
data = []
# iterate through rows
for i, txt in enumerate(df["Catalyst"].fillna("").astype(str)):
    # tokenize into sentences
    sentences = sent_tokenize(txt)
    # create entry for each row
    for j, s in enumerate(sentences):
        entry = {
            "id": f"biotech_{i}_{j}",
            "text": s
            }
        data.append(entry)

# write to json file
with open(output, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(data)} entries to {output}")
