# kNN Classifier for Embeddings
# Script for data classification based off semantic content 

### Setup
print("\nRunning setup...")

# Package import
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Set working directory
workdir = r"[WORK_DIR]"

# Read in labeled dataset
file = r"[LABELED_SET]"
df_ref = pd.read_csv(workdir + file)

# Read in test dataset
file = r"[TEST_SET]"
df_test = pd.read_csv(workdir + file)

# Stage labeled dataset
n = 100
def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

labs = df_ref[["Name", "Cat 1", "Cat 2", "Cat 3"]]
labs_flat = flatten(labs[["Name"]].values.tolist())
labs_trim = labs_flat#[:n] #For testing with n rows

# Stage test dataset
# print(df_test.columns)

test = df_test[["Name", "Cat 1", "Cat 2", "Cat 3"]]
test_flat = flatten(test[["Name"]].values.tolist())
test_trim = test_flat#[:n]

# Set up transformer for semantic embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 256


### Generate embeddings
print("\nGenerating embeddings...")
test_case_embedding = model.encode(labs_trim[0])
labs_embeddings = model.encode(labs_trim, normalize_embeddings=True)

### Get semantic similarity
k = 1 #Number of matches to generate
print("\nGetting semantic similarity...")

#Single-case test
#dot_scores = util.dot_score(test_case_embedding, labs_embeddings)[0]
#top_results = torch.topk(dot_scores, k = k)

#print("\nTop {k} matches for: {test_case}".format(k = k, test_case = labs_trim[0]))
#for score, idx in zip(top_results[0], top_results[1]):
#    print("\t" + labs_trim[idx], "\t(Score: {:.4f})".format(score))
#    print("\t\tCategories: " + labs.loc[labs["Name"] == labs_trim[idx]][['Cat 1', 'Cat 2', 'Cat 3']].to_string(index=False, header=False))

#Batch test
test_set_embeddings = model.encode(test_trim, normalize_embeddings=True)
out = pd.DataFrame(columns = ["Name", "Embedding Confidence", "Cat 1", "Cat 2", "Cat 3"])

for i in range(len(test_set_embeddings)):
    dot_scores = util.dot_score(test_set_embeddings[i], labs_embeddings)[0]
    top_results = torch.topk(dot_scores, k = k)

    print("\nTop {k} match(es) for: {test_case}".format(k = k, test_case = test_trim[i]))
    for score, idx in zip(top_results[0], top_results[1]):
        print("\t" + labs_trim[idx], "\t(Score: {:.4f})".format(score))
#        print("Categories: " + labs.loc[labs["Name"] == labs_trim[idx]][['Cat 1', 'Cat 2', 'Cat 3']])
        match = labs.loc[labs["Name"] == labs_trim[idx]][["Cat 1", "Cat 2", "Cat 3"]]
        out.loc[len(out.index)] = [test_trim[i], 
                                   "{:.4f}".format(score), 
                                   match["Cat 1"].values[0],
                                   match["Cat 2"].values[0],
                                   match["Cat 3"].values[0]]

print("\nPrinting output...")
print(out)

print("\nWriting to file...")
out.to_excel(workdir + r"\output.xlsx")

print("\nDone!")
