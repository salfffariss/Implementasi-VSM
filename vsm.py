import sys
import os
import math
import re
from nltk.stem import PorterStemmer

# Built-in stopwords (no download needed)
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','being','have','has','had','do','does',
    'did','will','would','could','should','may','might','shall','can','need',
    'that','this','these','those','it','its','i','you','he','she','we','they',
    'my','your','his','her','our','their','me','him','us','them','what','which',
    'who','how','when','where','why','not','no','nor','so','yet','both','either',
    'as','from','by','about','than','then','if','up','out','more','also','into',
    'through','during','before','after','above','below','between','each','few',
    'other','some','such','only','own','same','so','than','too','very','s','t'
}

stemmer = PorterStemmer()

def preprocess(text):
    # Lowercase, remove punctuation, tokenize, remove stopwords, stem
    text = text.lower()
    tokens = re.findall(r'[a-z]+', text)
    tokens = [stemmer.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 1]
    return tokens

def compute_tf(tokens):
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    tf = {}
    for t, f in freq.items():
        tf[t] = 1 + math.log10(f) if f > 0 else 0
    return tf

def main():
    if len(sys.argv) != 3:
        print("Usage: python vsm.py base.txt query.txt")
        sys.exit(1)

    base_file = sys.argv[1]
    query_file = sys.argv[2]

    # Read document list
    with open(base_file) as f:
        doc_names = [line.strip() for line in f if line.strip()]

    N = len(doc_names)

    # Preprocess all documents
    doc_tokens = {}
    for name in doc_names:
        with open(name) as f:
            doc_tokens[name] = preprocess(f.read())

    # Compute TF per doc
    doc_tf = {name: compute_tf(tokens) for name, tokens in doc_tokens.items()}

    # Compute IDF
    all_terms = set(t for tf in doc_tf.values() for t in tf)
    idf = {}
    for term in all_terms:
        n = sum(1 for tf in doc_tf.values() if term in tf)
        idf[term] = math.log10(N / n) if n > 0 else 0

    # Compute TF-IDF weights
    doc_weights = {}
    for name in doc_names:
        doc_weights[name] = {t: doc_tf[name][t] * idf[t] for t in doc_tf[name]}

    # Build inverted index: term -> list of (doc_num, weight)
    inverted = {}
    for i, name in enumerate(doc_names, 1):
        for term, w in doc_weights[name].items():
            if term not in inverted:
                inverted[term] = []
            inverted[term].append((i, round(w, 4)))

    # Write index.txt
    with open('index.txt', 'w') as f:
        for term in sorted(inverted):
            entries = ' '.join(f"{d},{w}" for d, w in inverted[term])
            f.write(f"{term}: {entries}\n")

    # Write weights.txt
    with open('weights.txt', 'w') as f:
        for name in doc_names:
            terms_str = ' '.join(f"{t}, {w:.4f}" for t, w in sorted(doc_weights[name].items()))
            f.write(f"{name}: {terms_str}\n")

    # Process query
    with open(query_file) as f:
        query_text = f.read()

    # Remove boolean operators
    query_text = re.sub(r'\b(AND|OR|NOT)\b', ' ', query_text)
    query_tokens = preprocess(query_text)
    query_tf = compute_tf(query_tokens)
    query_weights = {t: query_tf[t] * idf.get(t, 0) for t in query_tf}

    # Cosine similarity
    def cosine(vec1, vec2):
        dot = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in vec2)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

    scores = [(name, cosine(doc_weights[name], query_weights)) for name in doc_names]
    scores = [(n, s) for n, s in scores if s > 0.001]
    scores.sort(key=lambda x: -x[1])

    # Write response.txt
    with open('response.txt', 'w') as f:
        f.write(f"{len(scores)}\n")
        for name, score in scores:
            f.write(f"{name} {score:.4f}\n")

    print(f"Done! {len(scores)} relevant documents found.")
    for name, score in scores:
        print(f"  {name}: {score:.4f}")

if __name__ == '__main__':
    main()
