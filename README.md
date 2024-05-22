# Description 

This repository makes use of the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) package to load open-source large language models for embedding textual data. In `analysis/main.py`, the  `SentenceTransformer` class is used to import a model from [Hugging Face](https://huggingface.co/models) according to the `MODEL_PATH` constant in `analysis/config.py`.

Below the steps followed in `analysis/main.py` are described with most steps relying on functions defined in `analysis/utils.py` or in `analysis/database/db_utils.py`.


1. Fetch all company reviews or other textual data from SQLite database found in `analysis/database/data/` using `get_all_reviews()` function found in `analysis/database/db_utils.py`.

    - For company reviews, `reviews_dict` is created containing review ids as keys and review text as values.


2. Load model using the `SentenceTransformer` class from Sentence Transformers.


3. Generate embeddings for both extremes for each structural aspect such as centralized and decentralized for centralization. The `model.encode()` method is used to generate the embeddings with `normalize_embeddings=True` to normalize the values in the vectors returned. The examples of extremes reviews used in this step are found in `analysis/config.py` as lists of strings.


4. Calculate average embedding for each extreme using the `get_average_embedding()` function from `analysis/utils.py`.


5. Start a pool of workers for all available GPUs and assign each worker encoding tasks using `model.encode_multi_process()` method for all company reviews or other textual data. It is important to stop the multi-processing after processes are complete to clean up resources. 


6. Create a nested dictionary containing review ids as outer keys and an inner dictionary as the values containing the names of each structural aspect, such as centralization, as the inner keys and the similarity score of the review in terms of the given aspect as values for the inner dictionary. 

    - The `get_similarity_scores()` function is used to create the dictionary which calls the lower-level `get_similarity()` function that calculates the similarity measure of a vector with regards to two vectors representing the extremes of an a structural attribute such as centralized and decentralized for centralization of firms. The similarity measure is borrowed from Viashima and Samila, 2022 and ranges from -1 to 1, with 0 being neutral. 


7. The last step updates the database with the scores for each structural spectrum for each review or textual observation using the `update_reviews_scores()`. It adds columns for each structural aspect by the name, for example centralization.