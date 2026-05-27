import bm25s


# text_corpus = [
#     "a cat is a feline and likes to purr",
#     "a dog is the human's best friend and loves to play",
#     "a bird is a beautiful animal that can fly",
#     "a fish is a creature that lives in water and swims",
# ]
#
# metadata_corpus = [
#     {"id": "cat-doc", "title": "About Cat", "text": text_corpus[0]},
#     {"id": "dog-doc", "title": "About Dog", "text": text_corpus[1]},
#     {"id": "bird-doc", "title": "About Bird", "text": text_corpus[2]},
#     {"id": "fish-doc", "title": "About Fish", "text": text_corpus[3]},
# ]
#
# # Tokenize the corpus and only keep the ids (faster and saves memory)
# corpus_tokens = bm25s.tokenize(text_corpus, stopwords="en")
#
# # Create the BM25 model and index the corpus
# retriever = bm25s.BM25(corpus=metadata_corpus)
# retriever.index(corpus_tokens)
#
# # Query the corpus
# query = "does the fish purr like a cat?"
# query_tokens = bm25s.tokenize(query)
#
# # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
# # To return docs instead of IDs, set the `corpus=corpus` parameter.
# results, scores = retriever.retrieve(query_tokens, k=2)
#
# for i in range(results.shape[1]):
#     doc, score = results[0, i], scores[0, i]
#     print(f"Rank {i+1} (score: {score:.2f}): {doc}")
#
# # You can save the arrays to a directory...
# retriever.save("./animal_index_bm25")
#
# # You can save the corpus along with the model
# retriever.save("./animal_index_bm25", corpus=metadata_corpus)


# Load bm25 corpus retriever
print("Loading retriever")

query = "Electric guitar with Fishman Fluence active pickups and Floyd Rose floating tremolo."
query_tokens = bm25s.tokenize(query)

# reloaded_retriever = bm25s.BM25.load("./animal_index_bm25", load_corpus=True)  # load_corpus=True returns texts instead of indexes only
reloaded_retriever = bm25s.BM25.load("../src/data/bm25_cache/thomann_product_index_bm25", load_corpus=True)  # load_corpus=True returns texts instead of indexes only

results, scores = reloaded_retriever.retrieve(query_tokens, k=2)

print(results.squeeze().shape)
print(results.squeeze())
print(scores.squeeze().shape)
print(scores.squeeze())

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

