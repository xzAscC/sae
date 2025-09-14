import gensim.downloader as api
import numpy as np

def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def analogy_cos(model, a, b, c, d):
    """Compute cos(rep(a)-rep(b), rep(c)-rep(d))"""
    return cosine_sim(model[a] - model[b], model[c] - model[d])


print("Loading GoogleNews word2vec...")
# w2v = api.load("word2vec-google-news-300")

# GloVe (Common Crawl, 840B tokens, 300d)
print("Loading GloVe (Common Crawl)...")
glove = api.load("glove-wiki-gigaword-300")

pairs = ("man", "woman", "king", "queen")

print("\nCosine Similarity Results:")
# print("word2vec-google-news-300: ", analogy_cos(w2v, *pairs))
print("glove-wiki-gigaword-300: ", analogy_cos(glove, *pairs))
print(glove.most_similar(positive=["king", "woman"], negative=["man"], topn=10))
