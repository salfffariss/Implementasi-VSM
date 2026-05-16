# Vector Space Model (VSM) - Information Retrieval

**Mata Kuliah:** Aljabar Linear JTK POLBAN  
**Nama:** Muhammad Rizqi Sholahuddin

---

## Cara Menjalankan

```bash
pip install -r requirements.txt
python vsm.py base.txt query.txt
```

## Algoritma

1. **Preprocessing** — tokenisasi, hapus stopwords, stemming (Porter Stemmer)
2. **TF-IDF** — `TF = 1 + log(freq)`, `IDF = log(N/n)`
3. **Cosine Similarity** — `sim(d,q) = (d·q) / (|d|×|q|)`
4. **Ranking** — dokumen diurutkan dari similarity tertinggi

## Output

| File | Isi |
|------|-----|
| `index.txt` | Inverted index: `term: doc,bobot ...` |
| `weights.txt` | Bobot TF-IDF per dokumen |
| `response.txt` | Ranking dokumen berdasarkan query |

## Contoh Output (`response.txt`)

```
3
doc2.txt 0.9512
doc1.txt 0.4231
doc5.txt 0.0213
```
