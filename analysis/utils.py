from dask.dataframe import DataFrame as DaskDataFrame
from pandas import DataFrame
from numpy import dot, array, sum
from numpy.linalg import norm

from typing import List, Dict

from database import Review


def to_dask_dataframe(reviews: List[Review], npartitions: int = 10) -> DaskDataFrame:
    """
    Convert list of reviews to Dask DataFrame.
    """
    reviews_dict = [
        {
            "review_id": review.review_id,
            "employer_id": review.employer_id,
            "review_text": review.review_text,
        }
        for review in reviews
    ]

    df_reviews = DataFrame(reviews_dict)

    return DaskDataFrame.from_pandas(df_reviews, npartitions=npartitions)


def get_average_embedding(embeddings: List[List[float]]) -> list:
    """
    Get average embedding for group of related embeddings.
    """
    return [
        sum(dim_values) / len(dim_values)
        for dim_values in list(map(list, zip(*embeddings)))
    ]


def get_similarity(
    vector: List[float], pos_vector: List[float], neg_vector: List[float]
) -> float:
    """
    Get similarity measure from (Viashima and Samila, 2022).
    """
    vector_a, pos_vector_a, neg_vector_a = (
        array(vector),
        array(pos_vector),
        array(neg_vector),
    )
    cos_sim = (dot(vector_a - neg_vector_a, pos_vector_a - neg_vector_a)) / (
        norm(vector_a - neg_vector_a) * norm(pos_vector_a - neg_vector_a)
    )
    term = dot(cos_sim, norm(pos_vector_a - neg_vector_a)) / norm(
        pos_vector_a - neg_vector_a
    )

    return term - 0.5


def cosine_difference(
    vector: List[float], pos_vector: List[float], neg_vector: List[float]
) -> float:
    """
    Get cosine similarities for both pairs and take the difference.
    """
    vector_a, pos_vector_a, neg_vector_a = (
        array(vector),
        array(pos_vector),
        array(neg_vector),
    )
    cos_sim_pos = dot(vector_a, pos_vector_a) / (
        (norm(vector_a)) * (norm(pos_vector_a))
    )
    cos_sim_neg = dot(vector_a, neg_vector_a) / (
        (norm(vector_a)) * (norm(neg_vector_a))
    )

    return cos_sim_pos - cos_sim_neg


def get_similarity_scores(
    embeddings: List[List[float]],
    identifiers: List[int],
    positive_vectors: List[List[float]],
    negative_vectors: List[List[float]],
    positive_attributes: List[str],
) -> Dict[int, Dict[str, float]]:
    """
    Generate similarity scores dictionary for each document's embedding and each attribute's embedding.
    """
    return {
        id: {
            spec: get_similarity(vec, pos, neg)
            for pos, neg, spec in zip(
                positive_vectors, negative_vectors, positive_attributes
            )
        }
        for id, vec in zip(identifiers, embeddings)
    }


def get_cosdiff_scores(
    embeddings: List[List[float]],
    identifiers: List[int],
    pos_vectors: List[List[float]],
    neg_vectors: List[List[float]],
    pos_attributes: List[str],
) -> Dict[int, Dict[str, float]]:
    """
    Generate cosine difference scores dictionary for each document's embedding and each attribute's embedding.
    """
    return {
        id: {
            spec: cosine_difference(vec, pos, neg)
            for pos, neg, spec in zip(pos_vectors, neg_vectors, pos_attributes)
        }
        for id, vec in zip(identifiers, embeddings)
    }


def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """
    Get normalized embeddings.
    """
    return [vector / norm(vector) for vector in embeddings]
