import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Clean dataset
df = pd.read_csv('dataset/books.csv', on_bad_lines='skip')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df.drop('isbn13', axis=1)
df = df.drop('language_code', axis=1)
# Print dataset info
print("Shape:", df.shape)
print("\nFeatures:", df.columns)

# Scaling numerical features (ensure only numerical columns are scaled)
numerical_features = df[['num_pages', 'ratings_count', 'text_reviews_count']].select_dtypes(include=[np.number])

# Apply MinMax scaling to numerical features
scaler = MinMaxScaler()
scaled_numerical_features = scaler.fit_transform(numerical_features)

# Vectorize authors using HashingVectorizer
vectorizer = HashingVectorizer(n_features=50, analyzer='word', token_pattern=r"[^/]+")
author_vectors = vectorizer.fit_transform(df['authors'])

# Combine all features: numerical features and author vectors
combined_features = np.hstack([scaled_numerical_features, author_vectors.toarray()])

# Calculate cosine similarity
cos_sim_matrix = cosine_similarity(combined_features)

# Convert to DataFrame for easier viewing
cos_sim_df = pd.DataFrame(cos_sim_matrix, index=df['title'], columns=df['title'])

# Print the similarity matrix
print(cos_sim_df)

def recommend_books(book_title, n_recommendations=2):
    if book_title == '' or book_title == ' ':
        return "The name of book cant be empty string."
    if book_title not in df['title'].values:
        return f"Book titled '{book_title}' not found in the dataset."
    idx = df[df['title'] == book_title].index[0]
    
    similarity_scores = list(enumerate(cos_sim_matrix[idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    books = []
    recommended_indices = [i[0] for i in sorted_scores[1:n_recommendations+1]]
    return df.iloc[recommended_indices]['title'].tolist()

recommendations = recommend_books('Poor People')
print("Recommended books:", recommendations)    