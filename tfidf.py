import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHOICES = ("A", "B", "C", "D")


def load_corpus(corpus_file="dataset.json"):
    documents = []
    corpus_path = Path(corpus_file)

    try:
        with corpus_path.open("r", encoding="utf-8") as f:
            prefix = f.read(100).lstrip()
            f.seek(0)

            rows = json.load(f) if prefix.startswith("[") else f
            for line_num, row in enumerate(rows, 1):
                if prefix.startswith("["):
                    doc = row
                else:
                    line = row.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                if not isinstance(doc, dict):
                    continue

                metadata = doc.get("metadata") or {}
                parts = [
                    doc.get("title", ""),
                    doc.get("demuc_name", ""),
                    doc.get("chude_name", ""),
                    metadata.get("source_info", ""),
                    " ".join(map(str, metadata.get("cross_refs") or [])),
                    doc.get("content", ""),
                ]

                text_parts = []
                for part in parts:
                    if part is None:
                        continue
                    part = str(part).strip()
                    if part:
                        text_parts.append(part)

                text = " ".join(text_parts)
                if text:
                    documents.append(text)
    except json.JSONDecodeError:
        print(f"[LOI] File corpus khong dung dinh dang JSON: {corpus_path}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"[LOI] Khong tim thay file corpus: {corpus_path}")
        sys.exit(1)

    if not documents:
        print(f"[LOI] Corpus rong: {corpus_path}")
        sys.exit(1)

    print(f"[INFO] Da tai {len(documents):,} tai lieu tu corpus.")
    return documents


def make_submission(
    test_file="de_thi.json",
    corpus_file="dataset.json",
    output_file="submission.json",
):
    top_k = int(os.getenv("TOP_K", "5"))
    max_features = int(os.getenv("MAX_FEATURES", "500000"))

    documents = load_corpus(os.getenv("CORPUS_FILE", corpus_file))

    print("[INFO] Dang xay dung TF-IDF index...")
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    doc_vectors = vectorizer.fit_transform(documents)

    test_path = Path(os.getenv("TEST_FILE", test_file))
    try:
        with test_path.open("r", encoding="utf-8") as f:
            prefix = f.read(100).lstrip()
            f.seek(0)
            if prefix.startswith("["):
                test_data = json.load(f)
            else:
                test_data = [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError:
        print(f"[LOI] File de thi khong dung dinh dang JSON: {test_path}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"[LOI] Khong tim thay file de thi: {test_path}")
        sys.exit(1)

    submissions = []
    print(f"[INFO] Bat dau xu ly {len(test_data)} cau hoi...")

    for index, item in enumerate(test_data, 1):
        question = item.get("question", "")

        query_vector = vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        k = min(top_k, len(documents))
        top_doc_indices = np.argsort(similarities)[-k:][::-1]
        context = " ".join(documents[i].lower() for i in top_doc_indices)

        best_choice = "B"
        best_score = -1.0

        for choice_key in CHOICES:
            choice_text = str(item.get(choice_key, ""))
            choice_words = set(re.findall(r"\w+", choice_text.lower()))

            if choice_words:
                overlap = sum(1 for word in choice_words if word in context)
                overlap_score = overlap / len(choice_words)
            else:
                overlap_score = 0.0

            option_query = f"{question} {choice_text}".strip()
            option_vector = vectorizer.transform([option_query])
            option_scores = cosine_similarity(option_vector, doc_vectors).flatten()
            retrieval_score = float(option_scores[top_doc_indices].max())

            score = 0.85 * retrieval_score + 0.15 * overlap_score
            if score > best_score:
                best_score = score
                best_choice = choice_key

        submissions.append({
            "id": item.get("id"),
            "answer": best_choice,
        })

        if index % 50 == 0 or index == len(test_data):
            print(f"[INFO] Da xu ly {index}/{len(test_data)} cau hoi.")

    output_path = Path(output_file)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Da luu file ket qua: {output_path}")


if __name__ == "__main__":
    make_submission()
