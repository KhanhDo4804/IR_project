import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


CHOICES = ("A", "B", "C", "D")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
SECOND_BEST_RELATIVE_SCORE_THRESHOLD = float(os.getenv("SECOND_BEST_RELATIVE_SCORE_THRESHOLD", "0"))
NEGATIVE_PATTERNS = (
    r"\bkhông\s+thuộc\b",
    r"\bkhông\s+đúng\b",
    r"\bkhông\s+chính\s+xác\b",
    r"\bkhông\s+phải\b",
    r"\bkhông\s+bao\s+gồm\b",
    r"\bngoại\s+trừ\b",
    r"\btrừ\s+trường\s+hợp\b",
    r"\blà\s+sai\b",
    r"\bnhận\s+định\s+sai\b",
    r"\bphát\s+biểu\s+sai\b",
    r"\bphương\s+án\s+sai\b",
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def find_file(filename: str) -> Path:
    roots = [Path.cwd(), Path.cwd().parent, Path("/kaggle/working")]
    for root in roots:
        path = root / filename
        if path.exists():
            return path

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob(filename))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Cannot find {filename}")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def split_content(text: str) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    if CHUNK_WORDS <= 0 or len(words) <= CHUNK_WORDS:
        return [" ".join(words)]

    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    return [" ".join(words[start:start + CHUNK_WORDS]) for start in range(0, len(words), step)]


def document_chunks(doc: dict) -> list[str]:
    metadata = doc.get("metadata") or {}
    header_parts = [
        doc.get("id", ""),
        doc.get("title", ""),
        doc.get("demuc_name", ""),
        doc.get("chude_name", ""),
        metadata.get("source_info", ""),
        " ".join(map(str, metadata.get("cross_refs") or [])),
    ]
    header = " ".join(str(part) for part in header_parts if part)
    boosted_header = " ".join([header, header, header]).strip()

    chunks = []
    for content_chunk in split_content(doc.get("content", "")):
        text = f"{boosted_header} {content_chunk}".strip()
        if text:
            chunks.append(text)
    return chunks


def load_corpus(path: Path) -> list[str]:
    docs = []
    for doc in read_json(path):
        docs.extend(document_chunks(doc))
    if not docs:
        raise ValueError(f"No documents loaded from {path}")
    return docs


def is_negative_question(question: str) -> bool:
    question = f" {normalize(question)} "
    return any(re.search(pattern, question) for pattern in NEGATIVE_PATTERNS)


def top_score(scores, top_k: int) -> tuple[float, np.ndarray]:
    if hasattr(scores, "toarray"):
        scores = scores.toarray()
    scores = np.asarray(scores).ravel()
    k = min(top_k, len(scores))
    if k <= 0:
        return 0.0, np.array([], dtype=np.int64)

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    top_values = scores[top_idx]

    best = float(top_values[0])
    top3 = float(np.mean(top_values[: min(3, len(top_values))]))
    topk = float(np.mean(top_values))
    return 0.70 * best + 0.20 * top3 + 0.10 * topk, top_idx


def lexical_support(choice: str, context: str) -> float:
    choice = normalize(choice)
    words = set(word for word in choice.split() if len(word) > 1)
    if not words:
        return 0.0

    exact = 1.0 if len(choice) >= 8 and choice in context else 0.0
    coverage = sum(1 for word in words if word in context) / len(words)
    return 0.65 * coverage + 0.35 * exact


def choose_answer(item: dict, docs, doc_matrix, vectorizer, top_k: int) -> str:
    question = item.get("question", "")
    choices = {key: item.get(key, "") for key in CHOICES}

    question_vec = vectorizer.transform([question])
    _, question_top_idx = top_score(doc_matrix @ question_vec.T, top_k)
    question_context = normalize(" ".join(docs[i] for i in question_top_idx))

    option_queries = [f"{question} {choices[key]}" for key in CHOICES]
    option_vectors = vectorizer.transform(option_queries)
    option_scores = (doc_matrix @ option_vectors.T).toarray()

    scores = {}
    for idx, key in enumerate(CHOICES):
        tfidf_score, option_top_idx = top_score(option_scores[:, idx], top_k)
        context_idx = np.unique(np.concatenate([question_top_idx, option_top_idx]))
        option_context = normalize(" ".join(docs[i] for i in context_idx))
        lex_score = max(
            lexical_support(choices[key], question_context),
            lexical_support(choices[key], option_context),
        )
        scores[key] = 0.90 * tfidf_score + 0.10 * lex_score

    if is_negative_question(question):
        return min(scores, key=scores.get)

    best_answer = max(scores, key=scores.get)
    best_score = scores[best_answer]

    if SECOND_BEST_RELATIVE_SCORE_THRESHOLD > 0:
        ranked_answers = sorted(scores, key=scores.get, reverse=True)
        second_answer = ranked_answers[1]
        second_score = scores[second_answer]
        if second_score >= best_score * SECOND_BEST_RELATIVE_SCORE_THRESHOLD:
            return second_answer

    b_score = scores.get("B", 0.0)

    if b_score >= best_score * 0.9:
        return "B"

    return best_answer


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def write_submission(submission) -> Path:
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.json"

    with submission_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    return submission_path


def main():
    corpus_path = find_file(os.getenv("CORPUS_FILE", "dataset.json"))
    test_path = find_file(os.getenv("TEST_FILE", "de_thi.json"))
    top_k = int(os.getenv("TOP_K", "8"))
    max_features = int(os.getenv("MAX_FEATURES", "500000"))

    print(f"Corpus: {corpus_path}")
    print(f"Test set: {test_path}")
    print("Loading corpus...")
    docs = load_corpus(corpus_path)

    print(f"Loaded {len(docs)} documents. Building TF-IDF index...")
    vectorizer = build_vectorizer(max_features)
    doc_matrix = vectorizer.fit_transform(docs)

    questions = read_json(test_path)
    print(f"Answering {len(questions)} questions...")

    submission = []
    for idx, item in enumerate(questions, start=1):
        submission.append({
            "id": item.get("id"),
            "answer": choose_answer(item, docs, doc_matrix, vectorizer, top_k),
        })
        if idx % 50 == 0 or idx == len(questions):
            print(f"Done {idx}/{len(questions)}")

    submission_path = write_submission(submission)
    print(f"Wrote {submission_path}")


if __name__ == "__main__":
    main()
