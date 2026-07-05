"""
Pure-Python loading and tokenizing of the IMDB 50K Movie Reviews dataset.

No pandas, no scikit-learn — just the standard library (csv, string) plus
kagglehub for fetching the raw file, mirroring how Part 1 already fetches
its dataset in this repo.
"""

import csv
import os
import string

import kagglehub

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def download_dataset():
    """Download the Kaggle IMDB 50K CSV and return the path to it."""
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    for file_name in os.listdir(path):
        if file_name.endswith(".csv"):
            return os.path.join(path, file_name)
    raise FileNotFoundError("Could not find a CSV file in the downloaded dataset directory!")


def tokenize(review_text):
    """Lowercase, strip HTML line breaks and punctuation, return the set of unique words."""
    cleaned = review_text.replace("<br />", " ").lower()
    cleaned = cleaned.translate(_PUNCT_TABLE)
    return set(cleaned.split())


def load_reviews(csv_path):
    """Parse the CSV into a plain list of (word_set, label) tuples."""
    reviews = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append((tokenize(row["review"]), row["sentiment"]))
    return reviews
