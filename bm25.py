import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from rank_bm25 import BM25Okapi
TOP_K = 5


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())

class FastBM25Okapi:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_docs)
        self.doc_len = np.array([len(doc) for doc in tokenized_docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.vocab = {}

        postings = defaultdict(list)
        doc_freq = defaultdict(int)
        for doc_id, doc in enumerate(tokenized_docs):
            for term, tf in Counter(doc).items():
                postings[term].append((doc_id, tf))
                doc_freq[term] += 1

        rows = []
        cols = []
        data = []
        for term_id, (term, term_postings) in enumerate(postings.items()):
            self.vocab[term] = term_id
            df = doc_freq[term]
            idf = np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

            for doc_id, tf in term_postings:
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl
                )
                weight = idf * (tf * (self.k1 + 1) / denom)
                rows.append(doc_id)
                cols.append(term_id)
                data.append(weight)

        self.term_doc_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(self.n_docs, len(self.vocab)),
            dtype=float,
        ).tocsr()

    def get_scores(self, query_terms):
        if self.n_docs == 0 or not query_terms:
            return np.zeros(self.n_docs, dtype=float)

        query_freq = Counter(self.vocab[term] for term in query_terms if term in self.vocab)

        if not query_freq:
            return np.zeros(self.n_docs, dtype=float)

        query_vector = csr_matrix(
            (
                list(query_freq.values()),
                ([0] * len(query_freq), list(query_freq.keys())),
            ),
            shape=(1, len(self.vocab)),
            dtype=float,
        )
        return (query_vector @ self.term_doc_matrix.T).toarray().ravel()


def load_corpus(corpus_file):
    documents = []
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title=doc.get('title', '')
                title_token=re.split(r'[\s]+',title)
                title_parts=' '.join(title_token[2:])
                content=doc.get('content', '')
                chude_name=doc.get('chude_name', '')
                demuc_name=doc.get('demuc_name', '')
                text = f"{chude_name} {demuc_name} {title_parts} {content}".strip()
                if text:
                    documents.append(text)
    except FileNotFoundError:
        print(f"[ERROR] Cannot find corpus file: {corpus_file}")
        sys.exit(1)

    print(f"Loaded {len(documents):,} documents from corpus.")
    return documents


def load_test_data(test_file):
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Cannot find test file: {test_file}")
        sys.exit(1)
    print(f"Loaded {len(test_data)} questions from test set.")
    return test_data


def build_bm25_index(documents):
    print("Tokenizing corpus...")
    tokenized_docs = [
        tokenize(doc) for doc in documents
    ]

    print("Building global BM25 index...")
    bm25 = FastBM25Okapi(tokenized_docs)
    print(f"BM25 index ready. Documents: {len(tokenized_docs):,}")
    return bm25


def retrieve_top_k(query, bm25, documents, top_k=TOP_K):
    scores = bm25.get_scores(tokenize(query))

    k = min(top_k, len(scores))
    if k <= 0:
        return []

    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

    retrieved_docs = []
    for idx in top_k_indices:
        if scores[idx] > 0:
            retrieved_docs.append(documents[idx])

    return retrieved_docs


def select_answer(question_text, choices, context_text):
    valid_keys = ["A", "B", "C", "D"]

    if not context_text.strip():
        return "A"

    context_sentences = re.split(r"[.;!?\n]+", context_text)
    context_sentences = [sentence.strip() for sentence in context_sentences if sentence.strip()]
    if not context_sentences:
        return "A"

    tokenized_context = [tokenize(sentence) for sentence in context_sentences]
    tokenized_context = [tokens for tokens in tokenized_context if tokens]
    if not tokenized_context:
        return "A"

    tokenized_choices = []
    for key in valid_keys:
        choice_text = choices.get(key, "")
        candidate = f"{question_text} {choice_text}"
        tokenized_choices.append(tokenize(candidate))

    # Sử dụng thư viện rank_bm25 (xử lý âm IDF giống hệt code cũ)
    local_bm25 = BM25Okapi(tokenized_context)
    
    best_choice = "A"
    max_score = -1
    for key, tokenized_choice in zip(valid_keys, tokenized_choices):
        scores = local_bm25.get_scores(tokenized_choice)
        total_score = scores.sum()
        if total_score > max_score:
            max_score = total_score
            best_choice = key

    return best_choice


def make_submission(
    test_file="",
    corpus_file="",
    output_file="",
):
    documents = load_corpus(corpus_file)
    if not documents:
        print("Corpus is empty.")
        return

    bm25 = build_bm25_index(documents)

    test_data = load_test_data(test_file)

    submissions = []
    total = len(test_data)

    print(f"\nAnswering {total} questions...")
    print("=" * 60)

    for item in test_data:
        question_id = item.get("id")
        question_text = item.get("question", "")
        choices = {
            "A": item.get("A", ""),
            "B": item.get("B", ""),
            "C": item.get("C", ""),
            "D": item.get("D", ""),
        }

        full_query = f"{question_text} {choices['A']} {choices['B']} {choices['C']} {choices['D']}"
        retrieved_docs = retrieve_top_k(full_query, bm25, documents, top_k=TOP_K)
        context_text = " ".join(retrieved_docs) if retrieved_docs else ""
        best_answer = select_answer(question_text, choices, context_text)

        submissions.append(
            {
                "id": question_id,
                "answer": best_answer,
            }
        )

    print("=" * 60)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved results to: {output_file}")

    answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for sub in submissions:
        ans = sub.get("answer", "")
        if ans in answer_counts:
            answer_counts[ans] += 1

    print(f"\nTotal questions: {len(submissions)}")
    print(
        f"  Answer distribution: A={answer_counts['A']}, B={answer_counts['B']}, "
        f"C={answer_counts['C']}, D={answer_counts['D']}"
    )


if __name__ == "__main__":

    test_file="de_thi.json"
    corpus_file="dataset.json"
    
    output_file="submission.json"
    
    make_submission(
        test_file=str(test_file),
        corpus_file=str(corpus_file),
        output_file=str(output_file),
    )
