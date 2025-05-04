from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from typing import Optional
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import os

app = FastAPI()

# Global variables to cache after upload
big_data_cache = None
content_similarity_cache = None
recommdation2_cache = None

@app.post("/generate_recommendations/")
async def generate_recommendations(
        liked_books: UploadFile = File(...),

        #goodreads_interactions: UploadFile = File(...),
        #last_filter: UploadFile = File(...)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

) -> dict:
    global big_data_cache, content_similarity_cache, recommdation2_cache
    try:
        # Load data
        my_books = pd.read_csv(io.StringIO((await liked_books.read()).decode('utf-8')), on_bad_lines='skip')

        #goodreads_interactions_df = pd.read_csv(io.StringIO((await goodreads_interactions.read()).decode('utf-8')), delimiter=';', on_bad_lines='skip')
        #big_data = pd.read_csv(io.StringIO((await last_filter.read()).decode('utf-8')), delimiter=';', on_bad_lines='skip')
        file_path2 = os.path.join(BASE_DIR, "goodreads_interactions.csv")
        file_path1 = os.path.join(BASE_DIR, "last_filter.csv")

        big_data = pd.read_csv(file_path1, delimiter=';', on_bad_lines='skip')
        goodreads_interactions_df = pd.read_csv(file_path2, delimiter=';', on_bad_lines='skip')
        # Normalize column names
        for df in [my_books, goodreads_interactions_df, big_data]:
            df.columns = [col.strip().lower() for col in df.columns]

        # Column validation
        def check_columns(df, required, name):
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing column(s) {missing} in {name} file")

        check_columns(my_books, ['user_id', 'book_id', 'rating'], 'liked_books')
        check_columns(goodreads_interactions_df, ['user_id', 'book_id', 'rating'], 'goodreads_interactions')
        check_columns(big_data, ['book_id'], 'last_filter')  # only book_id required

        # Collaborative filtering
        my_books["book_id"] = my_books["book_id"].astype(str)
        book_set = set(my_books["book_id"])

        overlap_users = {}
        for index, row in goodreads_interactions_df.iterrows():
            user_id = str(row['user_id'])
            book_id = str(row['book_id'])
            if book_id in book_set:
                overlap_users[user_id] = overlap_users.get(user_id, 0) + 1

        filtered_users = {k for k, v in overlap_users.items() if v >= 1}

        interactions_list = []
        for index, row in goodreads_interactions_df.iterrows():
            user_id = str(row['user_id'])
            book_id = str(row['book_id'])
            rating = row['rating']
            if user_id in filtered_users:
                interactions_list.append([user_id, book_id, rating])

        interactions = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"])
        interactions = pd.concat([my_books[['user_id', 'book_id', 'rating']], interactions])

        # Matrix setup
        interactions["user_id"] = interactions["user_id"].astype(str)
        interactions["book_id"] = interactions["book_id"].astype(str)
        interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
        interactions.dropna(inplace=True)

        interactions["user_index"] = interactions["user_id"].astype("category").cat.codes
        interactions["book_index"] = interactions["book_id"].astype("category").cat.codes

        ratings_mat_coo = coo_matrix((interactions["rating"], (interactions["user_index"], interactions["book_index"])))
        ratings_mat = ratings_mat_coo.tocsr()

        my_index = 0
        user_similarity = cosine_similarity(ratings_mat[my_index, :], ratings_mat).flatten()

        indices = np.argpartition(user_similarity, -7)[-7:]
        similar_users = interactions[interactions["user_index"].isin(indices)]
        similar_users = similar_users[similar_users["user_id"] != "-1"]

        book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean'])
        book_recs = book_recs[(book_recs["mean"] >= 2) & (book_recs["count"] > 1)]

        recommended_books = book_recs.reset_index()["book_id"].astype(str).tolist()

        big_data_cache = big_data.copy()
        # Prepare content similarity cache (for the second endpoint)
        recommdation2 = big_data_cache.copy()
        recommdation2['tags'] = recommdation2['author'].fillna('')+ ' ' + recommdation2['description'].fillna('') + ' ' + recommdation2['categories'].fillna('')
        recommdation2 = recommdation2.drop(
            columns=['description', 'categories', 'author', 'book_pic', 'published_year', 'average_rate', 'num_pages'])

        cv = CountVectorizer(max_features=10000, stop_words='english')
        vector = cv.fit_transform(recommdation2['tags'].values.astype('U')).toarray()

        content_similarity = cosine_similarity(vector)
        recommdation2_cache = recommdation2
        content_similarity_cache = content_similarity

        return {
            "recommended_books": recommended_books,
            "message": "Collaborative recommendations generated successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")



@app.get("/recommend_by_book_id/")

def recommend_by_book_id(
    book_id: str = Query(..., description="Book ID for content-based recommendation"),
    top_n: int = 10
):
    global recommdation2_cache, content_similarity_cache

    # Ensure recommendations have been generated
    if recommdation2_cache is None or content_similarity_cache is None:
        raise HTTPException(status_code=400, detail="Please call /generate_recommendations/ first to upload and process the data.")

    try:
        # Normalize book_id and dataset
        book_id = book_id.strip()
        recommdation2_cache["book_id"] = recommdation2_cache["book_id"].astype(str).str.strip()

        # Validate book_id existence
        if book_id not in recommdation2_cache["book_id"].values:
            return {
                "recommended_books": [],
                "message": f"Book ID '{book_id}' not found in the dataset."
            }

        # Find index of requested book
        book_index = recommdation2_cache[recommdation2_cache["book_id"] == book_id].index[0]

        # Get similarity scores for this book
        similarity_scores = list(enumerate(content_similarity_cache[book_index]))

        # Sort by similarity, exclude the book itself (first entry)
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_similar_indices = [i[0] for i in sorted_scores if i[0] != book_index][:top_n]

        recommended_ids = recommdation2_cache.iloc[top_similar_indices]["book_id"].tolist()

        return {
            "recommended_books": recommended_ids,
            "message": f"Top {len(recommended_ids)} similar books to '{book_id}'"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during content-based recommendation: {str(e)}")
