#!/usr/bin/env python3
from pathlib import Path
import json
import re
import string

folder = Path("data")
folder.mkdir(exist_ok=True)

MEMORY_FILE = folder / "memory.json"

if MEMORY_FILE.exists():
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
else:
    memory = {}


def analyze_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    totals = [0, 0, 0, 0]
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        char_count = len(s.replace(" ", ""))
        word_count = len(s.split())
        punct_count = sum(1 for c in s if c in string.punctuation)
        alnum_count = sum(1 for c in s if c.isalnum())
        totals[0] += char_count
        totals[1] += word_count
        totals[2] += punct_count
        totals[3] += alnum_count
    num_sentences = len([s for s in sentences if s.strip()])
    if num_sentences == 0:
        return "0-0-0-0"
    averages = [round(t / num_sentences, 2) for t in totals]
    return f"{averages[0]}-{averages[1]}-{averages[2]}-{averages[3]}"


def parse_features(s):
    try:
        return [float(x) for x in s.split("-")]
    except ValueError:
        return None


def similarity_score(f1, f2):
    if len(f1) != len(f2):
        return -1
    diffs = [abs(a - b) for a, b in zip(f1, f2)]
    avg_diff = sum(diffs) / len(diffs)
    return 1 / (1 + avg_diff)

def predict(s):
    features = parse_features(s)
    if features is None:
        return None
    best_score = -1
    best_label = None
    if s in memory:
        return memory[s]
    for known_str, label in memory.items():
        known_features = parse_features(known_str)
        if known_features is None:
            continue
        score = similarity_score(features, known_features)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def get_multiline_input(prompt="Paste your text (end with empty line):"):
    print(prompt)
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    print("Welcome to AlgoDect! Type 'quit' to exit.")
    while True:
        raw_text = get_multiline_input()
        if raw_text.lower().strip() == "quit":
            break
        numeric_string = analyze_text(raw_text)
        guess = predict(numeric_string)
        if guess is None:
            print("No extracted features")
        else:
            print(f"Result: {guess.upper()}")
        if numeric_string not in memory:
            memory[numeric_string] = "unknown"
            with open(MEMORY_FILE, "w") as f:
                json.dump(memory, f)

if __name__ == "__main__":
    main()
