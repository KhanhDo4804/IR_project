import json, re
from collections import Counter, defaultdict
import numpy as np
from rank_bm25 import BM25Okapi
from scipy.sparse import coo_matrix, csr_matrix

TOP_K = 5
CHOICES = ("A", "B", "C", "D")


class FastBM25Okapi:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = np.array([len(doc) for doc in docs], dtype=float)
        self.avgdl = self.doc_len.mean() if self.n_docs else 0.0
        self.vocab = {}

        postings = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                postings[term].append((doc_id, tf))

        rows = []
        cols = []
        data = [] 
        for term_id, (term, hits) in enumerate(postings.items()):
            self.vocab[term] = term_id
            idf = np.log(1 + (self.n_docs - len(hits) + 0.5) / (len(hits) + 0.5))

            for doc_id, tf in hits:
                norm = 1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl if self.avgdl else 1
                rows.append(doc_id)
                cols.append(term_id)
                data.append(idf * tf * (k1 + 1) / (tf + k1 * norm))

        self.matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(self.n_docs, len(self.vocab)),
            dtype=float,
        ).tocsr()

    # Trả về điểm số BM25 cho một truy vấn đã được token hóa so với tất cả tài liệu
    def get_scores(self, terms):
        query = Counter(self.vocab[t] for t in terms if t in self.vocab)
        if not query:
            return np.zeros(self.n_docs, dtype=float)

        cols, vals = zip(*query.items())
        vector = csr_matrix(
            (vals, ([0] * len(cols), cols)),
            shape=(1, len(self.vocab)),
            dtype=float,
        )
        return (vector @ self.matrix.T).toarray().ravel()


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def load_corpus(corpus_file="dataset.json"):
    """Doc du lieu kien thuc (knowledge base) tu file JSON."""
    documents = []
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Ghep title va content lam noi dung de tim kiem
                title = " ".join(str(doc.get("title", "")).split()[2:])
                text = " ".join(str(x).strip() for x in [
                    doc.get("chude_name", ""),
                    doc.get("demuc_name", ""),
                    title,
                    doc.get("content", "")
                ] if x)
                if text:
                    documents.append(text)
    except FileNotFoundError:
        print(f"Loi: Khong tim thay file {corpus_file}. Vui long kiem tra lai!")
    return documents

def top_docs(query, fast_bm25, docs, k=TOP_K):
    scores = fast_bm25.get_scores(tokenize(query))
    k = min(k, len(scores))
    if k == 0:
        return []

    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [docs[i] for i in idx if scores[i] > 0]


def choose_answer(question, choices, context):
    sentences = [s.strip() for s in re.split(r"[.;!?\n]+", context) if s.strip()]
    tokenized = [tokens for tokens in map(tokenize, sentences) if tokens]
    if not tokenized:
        return "B"

    local_bm25 = BM25Okapi(tokenized)
    best_score = -1
    best_choice = "B"
    for key in CHOICES:
        query = f"{question} {choices[key]}"
        query_tokens = tokenize(query)

        scores = local_bm25.get_scores(query_tokens)
        total_score = scores.sum()

        if total_score > best_score:
            best_score = total_score
            best_choice = key

    return best_choice


def make_submission(test_file, corpus_file, output_file):
    docs = load_corpus(corpus_file)
    bm25 = FastBM25Okapi([tokenize(doc) for doc in docs])

    with open(test_file, encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    total = len(questions)
    for idx, item in enumerate(questions, 1):
        question = item.get("question", "")
        choices = {key: item.get(key, "") for key in CHOICES}
        query = " ".join([question] + [choices[key] for key in CHOICES])
        context = " ".join(top_docs(query, bm25, docs))
        rows.append({"id": item.get("id"), "answer": choose_answer(question, choices, context)})
        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total} questions")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} answers to {output_file}")
    print("Answer distribution:", dict(Counter(row["answer"] for row in rows)))

if __name__ == "__main__":
    corpus_file = "dataset.json"
    test_file="de_thi.json"
    output_file="submission.json"
    make_submission(test_file, corpus_file, output_file)
