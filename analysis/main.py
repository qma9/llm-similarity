from sentence_transformers import SentenceTransformer

from datetime import timedelta
from time import perf_counter

from log import setup_logging, logger
from database import get_all_reviews, update_reviews_scores
from utils import (
    get_average_embedding,
    get_similarity_scores,
)
from config import (
    MODEL_PATH,
    MODEL,
    POSITIVE_ATTRIBUTES,
    CENTRALIZED_REVIEWS,
    DECENTRALIZED_REVIEWS,
    SECRETIVE_REVIEWS,
    TRANSPARENT_REVIEWS,
    HIERARCHICAL_REVIEWS,
    NON_HIERARCHICAL_REVIEWS,
    FORMAL_REVIEWS,
    INFORMAL_REVIEWS,
    STAGNATING_REVIEWS,
    INNOVATIVE_REVIEWS,
    RISK_AVERSE_REVIEWS,
    RISK_TAKING_REVIEWS,
)


def main():
    total_start_time = perf_counter  # Benchmark total runtime

    # Get all reviews from SQLite database
    reviews = get_all_reviews()
    logger.info(f"Total reviews: {len(reviews)}", extra={"total_reviews": len(reviews)})

    # Create a dictionary with review_id as the key and review_text as the value
    reviews_dict = {review.review_id: review.review_text for review in reviews}

    # Import model
    model = SentenceTransformer(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}.")

    # Generate embeddings for both extremes of structural aspects
    centralized_embeddings = model.encode(
        CENTRALIZED_REVIEWS, normalize_embeddings=True
    )
    decentralized_embeddings = model.encode(
        DECENTRALIZED_REVIEWS, normalize_embeddings=True
    )
    secretive_embeddings = model.encode(SECRETIVE_REVIEWS, normalize_embeddings=True)
    transparent_embeddings = model.encode(
        TRANSPARENT_REVIEWS, normalize_embeddings=True
    )
    hierarchical_embeddings = model.encode(
        HIERARCHICAL_REVIEWS, normalize_embeddings=True
    )
    non_hierarchical_embeddings = model.encode(
        NON_HIERARCHICAL_REVIEWS, normalize_embeddings=True
    )
    formal_embeddings = model.encode(FORMAL_REVIEWS, normalize_embeddings=True)
    informal_embeddings = model.encode(INFORMAL_REVIEWS, normalize_embeddings=True)
    stagnating_embeddings = model.encode(STAGNATING_REVIEWS, normalize_embeddings=True)
    innovative_embeddings = model.encode(INNOVATIVE_REVIEWS, normalize_embeddings=True)
    risk_averse_embeddings = model.encode(
        RISK_AVERSE_REVIEWS, normalize_embeddings=True
    )
    risk_taking_embeddings = model.encode(
        RISK_TAKING_REVIEWS, normalize_embeddings=True
    )
    logger.info("Encoded extreme reviews.")

    # Get average embeddings for attribute extremes
    centralized_average = get_average_embedding(centralized_embeddings)
    decentralized_average = get_average_embedding(decentralized_embeddings)
    secretive_average = get_average_embedding(secretive_embeddings)
    transparent_average = get_average_embedding(transparent_embeddings)
    hierarchical_average = get_average_embedding(hierarchical_embeddings)
    non_hierarchical_average = get_average_embedding(non_hierarchical_embeddings)
    formal_average = get_average_embedding(formal_embeddings)
    informal_average = get_average_embedding(informal_embeddings)
    stagnating_average = get_average_embedding(stagnating_embeddings)
    innovative_average = get_average_embedding(innovative_embeddings)
    risk_averse_average = get_average_embedding(risk_averse_embeddings)
    risk_taking_average = get_average_embedding(risk_taking_embeddings)
    logger.info("Calculated average embeddings for extremes.")

    try:

        # Start the multi-process pool on all available CUDA devices
        pool = model.start_multi_process_pool()

        model_start_time = perf_counter()  # Benchmark model runtime

        # Generate embeddings for all reviews
        review_embeddings = model.encode_multi_process(reviews_dict.values(), pool)

        model_end_time = perf_counter()  # Benchmark model runtime

    finally:

        # Optional: Stop the processes in the pool
        model.stop_multi_process_pool(pool)

    # Log model runtime
    model_time = timedelta(seconds=(model_end_time - model_start_time))
    logger.info(
        f"{MODEL} model ran in: {model_time}",
        extra={"model": MODEL, "time": model_time},
    )

    # Get similarity scores for review embeddings
    similarity_scores_dict = get_similarity_scores(
        embeddings=review_embeddings,
        identifiers=reviews_dict.keys(),
        positive_vectors=[
            centralized_average,
            secretive_average,
            hierarchical_average,
            formal_average,
            stagnating_average,
            risk_averse_average,
        ],
        negative_vectors=[
            decentralized_average,
            transparent_average,
            non_hierarchical_average,
            informal_average,
            innovative_average,
            risk_taking_average,
        ],
        positive_attributes=POSITIVE_ATTRIBUTES,
    )

    # Update reviews with similarity scores
    update_reviews_scores(reviews, similarity_scores_dict)

    total_end_time = perf_counter  # Benchmark total runtime

    # Log total runtime
    total_time = timedelta(seconds=(total_end_time - total_start_time))
    logger.info(f"Total runtime: {total_time}", extra={"total_time": total_time})


if __name__ == "__main__":

    # Setup logging
    listener = setup_logging()

    main()

    # Stop listener
    listener.stop()
