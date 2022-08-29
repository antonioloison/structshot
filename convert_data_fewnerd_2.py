
import sys
import json

sys.path.append("/content/ml-lab/014_few-shot-ner/")


import sys
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split


fewnerd_dataset = []
with open("data/fewnerd/big_train.txt", "r") as f:
    tokens = []
    tags = []
    for line in f.readlines():
        if len(line) > 1:
            token, tag = line.split("\t")
            token, tag = token.strip(), tag.strip()

            tokens.append(token)
            tags.append(tag)
        else:
            fewnerd_dataset.append({"tokens": tokens, "tags": tags})
            tokens, tags = [], []

train_dataset, test_dataset = train_test_split(fewnerd_dataset, test_size=0.1, random_state=10)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=10)

print("len(train_dataset)", len(train_dataset))
print("len(val_dataset)", len(val_dataset))
print("len(test_dataset)", len(test_dataset))

for dataset, filename in zip([train_dataset, val_dataset, test_dataset],
                             ["data/fewnerd/train.txt",
                              "data/fewnerd/dev.txt",
                              "data/fewnerd/test.txt"]):
    with open(filename, "w") as f:
        for example in dataset:
            tokens, tags = example["tokens"], example["tags"]
            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")
