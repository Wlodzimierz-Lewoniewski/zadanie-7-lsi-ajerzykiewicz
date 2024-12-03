import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(text):
    return [word.strip() for word in re.split(r'[\.,\?!: ]+', text.lower()) if word]

def term_doc_matrix(docs, vocab):
    matrix = np.zeros((len(vocab), len(docs)))
    for j, doc in enumerate(docs):
        words = tokenize(doc)
        for i, term in enumerate(vocab):
            matrix[i, j] = 1 if term in words else 0
    return matrix

def lsi():
    print("Podaj liczbę dokumentów:")
    n = int(input())

    print("Podaj dokumenty")
    docs = [input().strip() for _ in range(n)]

    print("Podaj zapytanie:")
    query = input().strip()

    print("Podaj liczbę wymiarów po redukcji:")
    k = int(input())

    vocab = sorted(set(term for doc in docs + [query] for term in tokenize(doc)))
    query_terms = tokenize(query)

    matrix = term_doc_matrix(docs, vocab)

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    docs_reduced = S_k @ Vt_k

    query_vec = np.zeros(len(vocab))
    for term in query_terms:
        if term in vocab:
            query_vec[vocab.index(term)] = 1

    query_reduced = np.linalg.inv(S_k) @ (U_k.T @ query_vec)

    sims = cosine_similarity(query_reduced.reshape(1, -1), docs_reduced.T).flatten()

    rounded_sims = [round(sim, 2) for sim in sims]

    print(f"[{', '.join(map(str, rounded_sims))}]")

lsi()
