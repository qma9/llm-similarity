import numpy as np
from numpy import dot, array
from numpy.linalg import norm

def get_average_embedding(embeddings: list[list[float]]) -> list:
    """Get average embedding for group of related embeddings"""
    return [np.sum(dim_values) / len(dim_values) for dim_values in list(map(list, zip(*embeddings)))]


# def get_similarity(vector: list[float], pos_vector: list[float], neg_vector: list[float]) -> float:
#     """
#     Similarity function
#     Viashima and Samila, 2022
#     """
#     vector_a, pos_vector_a, neg_vector_a = array(vector), array(pos_vector), array(neg_vector)
#     cos_sim = (dot(vector_a - neg_vector_a, pos_vector_a - neg_vector_a)) / (norm(vector_a - neg_vector_a) * norm(pos_vector_a - neg_vector_a))
#     term = (cos_sim * norm(pos_vector_a - neg_vector_a)) / norm(pos_vector_a - neg_vector_a)

#     return term - 0.5


def get_similarity(vector: list[float], pos_vector: list[float], neg_vector: list[float]) -> float:
    """
    Similarity function
    Cao, Koning, and Nanda (Revised 2023)
    """
    vector_a, pos_vector_a, neg_vector_a = array(vector), array(pos_vector), array(neg_vector)
    cos_sim = (dot(vector_a - neg_vector_a, pos_vector_a - neg_vector_a)) / (norm(vector_a - neg_vector_a) * norm(pos_vector_a - neg_vector_a))
    term = (cos_sim * norm(vector_a - neg_vector_a)) / norm(pos_vector_a - neg_vector_a)

    return term - 0.5

def cosine_difference(vector: list[float], pos_vector: list[float], neg_vector: list[float]) -> float:
    """Gets cosine similarities for both pairs and takes the difference"""
    vector_a, pos_vector_a, neg_vector_a = array(vector), array(pos_vector), array(neg_vector)
    cos_sim_pos = dot(vector_a, pos_vector_a) / ((norm(vector_a)) * (norm(pos_vector_a)))
    cos_sim_neg = dot(vector_a, neg_vector_a) / ((norm(vector_a)) * (norm(neg_vector_a)))
    
    return cos_sim_pos - cos_sim_neg

# def get_similarity(vector: list[float], pos_vector: list[float], neg_vector: list[float]) -> float:
#     """
#     Similarity function
#     Cao, Koning, and Nanda (Revised 2023)
#     Version 2
#     """
#     vector_a, pos_vector_a, neg_vector_a = array(vector), array(pos_vector), array(neg_vector)
#     term = dot(pos_vector_a - neg_vector_a, vector_a - neg_vector_a) / norm(pos_vector_a - neg_vector_a)**2

#     return term - 0.5


def get_similarity_scores(embeddings: list[list[float]], pos_vectors: list[list[float]], neg_vectors: list[list[float]], pos_attributes: list[str]) -> dict:
    """Generate similarity scores dictionary for each document's embedding and each attribute's embedding"""
    return {i: {spec: get_similarity(vec, pos, neg) for pos, neg, spec in zip(pos_vectors, neg_vectors, pos_attributes)} for i, vec in enumerate(embeddings)}


def get_cosdiff_scores(embeddings: list[list[float]], pos_vectors: list[list[float]], neg_vectors: list[list[float]], pos_attributes: list[str]) -> dict:
    """Generate similarity scores dictionary for each document's embedding and each attribute's embedding"""
    return {i: {spec: cosine_difference(vec, pos, neg) for pos, neg, spec in zip(pos_vectors, neg_vectors, pos_attributes)} for i, vec in enumerate(embeddings)}


def normalize_embeddings(embeddings: list[list[float]]) -> list[list[float]]:
    """Get normalized embeddings"""
    return [vector / norm(vector) for vector in embeddings] 


if __name__ == '__main__':
    pass