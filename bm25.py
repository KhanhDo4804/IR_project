import heapq
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


CHOICES = ("A", "B", "C", "D")

COMMON_WORDS = {
    "thì", "và", "ở", "là", "của", "cho",  "ra", "trong", "được", "theo", "từ", "đến", "do", "bị", 
    "một", "các", "những", "về", "khi", "này", "đó", "tại", "với","để", "có", "làm", "việc",
}

class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.texts = []
        self.token_sets = []
        self.number_sets = []
        self.doc_lengths = []
        self.postings = defaultdict(list)
        self.idf = {}
        self.avgdl = 0.0
        self.n_docs = 0
        self._build(chunks)

    def _build(self, chunks):
        total_len = 0
        df = defaultdict(int)

        for doc_idx, chunk in enumerate(chunks):
            text = chunk["text"]
            counts = Counter(tokenize(text))

            self.texts.append(text)
            self.token_sets.append(set(counts))
            self.number_sets.append(extract_numbers(text))
            self.doc_lengths.append(sum(counts.values()))
            total_len += self.doc_lengths[-1]

            for term, tf in counts.items():
                self.postings[term].append((doc_idx, tf))
                df[term] += 1

        self.n_docs = len(self.texts)
        self.avgdl = total_len / self.n_docs if self.n_docs else 0.0
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query, top_n=20):
        scores = defaultdict(float)
        terms = dict.fromkeys(tokenize(query, remove_stopwords=True))

        for term in terms:
            for doc_idx, tf in self.postings.get(term, []):
                doc_len = self.doc_lengths[doc_idx]
                norm = 1 - self.b + self.b * doc_len / self.avgdl if self.avgdl else 1
                scores[doc_idx] += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + self.k1 * norm)

        return heapq.nlargest(top_n, scores.items(), key=lambda item: item[1])
    

def normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"\b0+(\d+)\b", r"\1", text)
    text = text.replace("_", " ")
    text = " ".join(re.findall(r"\w+", text, flags=re.UNICODE))
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text, remove_stopwords=False):
    tokens = normalize(text).split()
    if remove_stopwords:
        return [token for token in tokens if token not in COMMON_WORDS]
    return tokens


def extract_numbers(text):
    return set(re.findall(r"\d+", str(text or "")))


def load_corpus(corpus_file="dataset.json"):
    """Doc du lieu kien thuc (knowledge base) tu file JSON."""
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
                # Ghep title va content lam noi dung de tim kiem
                title = " ".join(str(doc.get("title", "")).split()[2:])
                text = " ".join(str(x).strip() for x in [
                    doc.get("chude_name", ""),
                    doc.get("demuc_name", ""),
                    title,
                    doc.get("content", ""),
                ] if x)
                if text:
                    documents.append(text)
    except FileNotFoundError:
        print(f"Loi: Khong tim thay file {corpus_file}. Vui long kiem tra lai!")

    return documents


def make_chunks(documents, max_tokens=180):
    chunks = []
    for doc_id, document in enumerate(documents):
        paragraphs = [normalize(part) for part in re.split(r"\n\s*\n+", document) if part.strip()]

        current = []
        current_len = 0
        for paragraph in paragraphs:
            tokens = tokenize(paragraph)
            if not tokens:
                continue

            if current and current_len + len(tokens) > max_tokens:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{len(chunks)}",
                    "text": " ".join(current),
                })
                current = []
                current_len = 0

            current.append(paragraph)
            current_len += len(tokens)

        if current:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{len(chunks)}",
                "text": " ".join(current),
            })

    return chunks


def score_option(option, hits, index):
    option_text = normalize(option)
    option_terms = set(tokenize(option_text, remove_stopwords=True))
    option_numbers = extract_numbers(option_text)
    if not option_terms and not option_numbers:
        return 0.0

    total = 0.0
    for rank, (doc_idx, bm25_score) in enumerate(hits, 1):
        chunk_text = index.texts[doc_idx]
        overlap = len(option_terms & index.token_sets[doc_idx]) / max(1, len(option_terms))
        number_score = len(option_numbers & index.number_sets[doc_idx]) / max(1, len(option_numbers)) if option_numbers else 0
        exact_bonus = 1.5 if option_text and option_text in chunk_text else 0.0
        rank_weight = 1 / math.log2(rank + 1)
        total += rank_weight * (bm25_score * (0.65 + 0.35 * overlap) + 0.8 * number_score + exact_bonus)

    return total


def answer_question(question, index, cache, top_k=20):
    q_text = question.get("question", "")
    expanded = " ".join([q_text] + [question.get(choice, "") for choice in CHOICES])

    q_hits = cache.setdefault(normalize(q_text), index.search(q_text, top_k))
    expanded_hits = cache.setdefault(normalize(expanded), index.search(expanded, top_k))
    merged = {}
    for weight, hits in ((1.05, q_hits), (1.0, expanded_hits)):
        for doc_idx, score in hits:
            merged[doc_idx] = max(merged.get(doc_idx, 0.0), score * weight)
    hits = heapq.nlargest(top_k, merged.items(), key=lambda item: item[1])

    if not hits:
        return "A"

    scores = {
        choice: score_option(question.get(choice, ""), hits, index)
        for choice in CHOICES
    }

    if all(score == 0 for score in scores.values()):
        scores = {
            choice: sum(score for _, score in index.search(f"{q_text} {question.get(choice, '')}", top_n=5))
            for choice in CHOICES
        }

    return max(scores, key=scores.get)


def make_submission(
    test_file="de_thi.json",
    corpus_file="dataset.json",
    output_file="submission.json",
    zip_file="submission.zip",
    top_k=10,
):  
    documents = load_corpus(corpus_file)
    print("Loaded corpus with {} documents.".format(len(documents)))
    if not documents:
        print("Corpus is empty.")
        return

    chunks = make_chunks(documents)
    print(f"Built {len(chunks)} chunks.")
    index = BM25Index(chunks)

    with open(test_file, encoding="utf-8") as f:
        questions = json.load(f)

    cache = {}
    predictions = []
    total = len(questions)

    for idx, question in enumerate(questions, 1):
        predictions.append({
            "id": question.get("id"),
            "answer": answer_question(question, index, cache, top_k=top_k),
        })
        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total} questions")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, arcname=Path(output_file).name)
        if "__file__" in globals() and Path(__file__).exists():
            zf.write(__file__, arcname=Path(__file__).name)

    print(f"Done: {output_file}, {zip_file}")


if __name__ == "__main__":
    make_submission()
