
import sys

sys.path.append("/content/ml-lab/014_few-shot-ner/")

# Commented out IPython magic to ensure Python compatibility.
# #%%script false
# %%capture
# !pip install transformers datasets jsonlines seqeval
# !pip install pyyaml==5.4.1

from collections import Counter, defaultdict
import os
import random
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# from evaluation import label_semantics_evaluation, label_semantics_display_preds, get_bio_litteral_label
# from preprocessing import ONTONOTES_LABELS, load_conll, load_ontonotes, preprocess_labels, process_dataset, random_selection

# Commented out IPython magic to ensure Python compatibility.
# %cd structshot

fin_train_dataset = []
fin_test_dataset = []

indexes = {"O": 0, "PER": 1, "ORG": 2, "LOC": 3, "MISC": 4}

for filename, dataset in zip(["dataset/FIN5.txt",
                              "dataset/FIN3.txt"],
                             [fin_train_dataset, fin_test_dataset]):
    with open(filename, "r") as f:
        tokens = []
        tags = []
        id = -1
        for line in tqdm(f.readlines()):
            if len(line) == 1:
                id += 1
                new_tags = []
                for i, tag in enumerate(tags):
                    new_tag = tag.strip()
                    if new_tag != "O":
                        if (i == 0 or tags[i-1] == "O" or tags[i-1][2] != tag[2]) and tag[0] == "I":
                            new_tag = new_tag.replace("I-", "B-")
                            new_tags.append(2 * indexes[new_tag.split("-")[1]] - 1)
                        else:
                            new_tags.append(2 * indexes[new_tag.split("-")[1]])
                    else:
                        new_tags.append(0)
                dataset.append(
                    {
                        "id": id,
                        "tokens": tokens,
                        "tags": new_tags,
                    }
                )
                tokens = []
                tags = []
            else:
                splitted_line = line.split()
                tokens.append(splitted_line[0])
                tags.append(splitted_line[-1])

fin_train_raw_dataset = fin_train_dataset[1:]
fin_test_raw_dataset = fin_test_dataset[1:]

print("Length of training dataset", len(fin_train_raw_dataset))
print("Max token length", max([len(x["tokens"]) for x in fin_train_raw_dataset]))
print("Length of test dataset", len(fin_test_raw_dataset))
print("Max token length", max([len(x["tokens"]) for x in fin_test_raw_dataset]))

def random_selection(dataset, labels, K, seed=10):
    random.seed(seed)
    counts = defaultdict(int)
    final_counts = defaultdict(int)
    support_set = []
    added_ids = set()
    for label in labels:
        if counts[label] < K:
            dataset_samples = [
                x
                for x in dataset
                if ((label * 2) in x["tags"] or (label * 2 - 1) in x["tags"])
                and x["id"] not in added_ids
            ]
            random.shuffle(dataset_samples)
            split_dataset = dataset_samples[: (K - counts[label])]
            for sample in split_dataset:
                added_ids.add(sample["id"])
                checked_tags = set()
                for tag in sample["tags"]:
                    refactored_tag = (tag + 1) // 2
                    if refactored_tag not in checked_tags:
                        counts[refactored_tag] += 1
                        checked_tags.add(refactored_tag)
                        final_counts[tag] += 1
            support_set += split_dataset
    assert len(added_ids) == len(support_set)
    # print(counts)
    # for tag in counts.keys():
    #     assert counts[tag] >= K
    return support_set

fin_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

def process_fin_dataset(support_dataset, query_dataset):
    final_dataset = {"support": {"word": [], "label": []}, "query": {"word": [], "label": []}, "types": ["person", "organization", "location", "miscellaneous", "noclass"]}
    for dataset, set_type in zip([support_dataset, query_dataset],
                           ["support", "query"]):
        for x in dataset:
            final_dataset[set_type]["word"].append(x["tokens"])
            new_tags = [fin_labels[numeric_tag] for numeric_tag in x["tags"]]
            final_dataset[set_type]["label"].append(new_tags)
    return final_dataset


final_datasets = []
for i in range(5):
    fin_loaded_dataset = random_selection(
        fin_train_raw_dataset, list(range(1, 5)), 5, seed=i
    )
    final_fin_dataset = process_fin_dataset(fin_loaded_dataset, fin_test_raw_dataset)
    final_datasets.append(final_fin_dataset["support"])

if not(os.path.exists("data/support-fin-5shot")):
    os.makedirs("data/support-fin-5shot")

final_datasets[0].keys()

for i, dataset in enumerate(final_datasets):
    tokens = dataset["word"]
    tags = dataset["label"]
    with open(f"data/support-fin-5shot/{i}.txt", "w") as f:
        for token_list, tag_list in zip(tokens, tags):
            for token, tag in zip(token_list, tag_list):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

final_datasets = []
for i in range(5):
    fin_loaded_dataset = random_selection(
        fin_train_raw_dataset, list(range(1, 5)), 5, seed=i
    )
    final_fin_dataset = process_fin_dataset(fin_loaded_dataset, fin_test_raw_dataset)
    final_datasets.append(final_fin_dataset["support"])

if not(os.path.exists("data/support-fin-1shot")):
    os.makedirs("data/support-fin-1shot")

final_datasets[0].keys()

for i, dataset in enumerate(final_datasets):
    tokens = dataset["word"]
    tags = dataset["label"]
    with open(f"data/support-fin-1shot/{i}.txt", "w") as f:
        for token_list, tag_list in zip(tokens, tags):
            for token, tag in zip(token_list, tag_list):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")


print(fin_train_raw_dataset[0])

training_set, dev_set = train_test_split(fin_train_raw_dataset)

train_tokens_list = [x["tokens"] for x in training_set]
train_tags_list = [x["tags"] for x in training_set]
dev_tokens_list = [x["tokens"] for x in dev_set]
dev_tags_list = [x["tags"] for x in dev_set]
test_tokens_list = [x["tokens"] for x in fin_test_raw_dataset]
test_tags_list = [x["tags"] for x in fin_test_raw_dataset]

with open("data/train-fin.txt", "w") as f:
    for tokens, tags in zip(train_tokens_list, train_tags_list):
        new_tags = [fin_labels[numeric_tag] for numeric_tag in tags]
        for token, tag in zip(tokens, new_tags):
            f.write(f"{token}\t{tag}\n")
        f.write("\n")

with open("data/dev-fin.txt", "w") as f:
    for tokens, tags in zip(dev_tokens_list, dev_tags_list):
        new_tags = [fin_labels[numeric_tag] for numeric_tag in tags]
        for token, tag in zip(tokens, new_tags):
            f.write(f"{token}\t{tag}\n")
        f.write("\n")

with open("data/test-fin.txt", "w") as f:
    for tokens, tags in zip(test_tokens_list, test_tags_list):
        new_tags = [fin_labels[numeric_tag] for numeric_tag in tags]
        for token, tag in zip(tokens, new_tags):
            f.write(f"{token}\t{tag}\n")
        f.write("\n")
