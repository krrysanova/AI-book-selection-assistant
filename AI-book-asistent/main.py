import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz import process

app = Flask(__name__)

# Cleaning dataset
df = pd.read_csv('dataset/books.csv', on_bad_lines='skip')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df.drop('isbn13', axis=1)
df = df.drop('language_code', axis=1)

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
cos_sim_df = pd.DataFrame(cos_sim_matrix, index=df['title'], columns=df['title'])

@app.route('/')
def home():
    return "Welcome to the Book Recommendation API!"

def recommend_books(book_title):
    if book_title == '' or book_title == ' ':
        return "The name of book cant be empty string."
    if book_title not in df['title'].values:
        return f"Book titled '{book_title}' not found in the dataset."
    idx = df[df['title'] == book_title].index[0]
    
    similarity_scores = list(enumerate(cos_sim_matrix[idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    books = []
    recommended_indices = [i[0] for i in sorted_scores[1:11]]
    return df.iloc[recommended_indices]['title'].tolist()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()  # Отримуємо JSON з тіла запиту
    book_title = data.get('book_title')  # Отримуємо назву книги

    if not book_title:
        return jsonify({"error": "Book title is required!"}), 400

    recommendations = recommend_books(find_closest_book(book_title))  # Викликаємо функцію для рекомендацій
    return jsonify({"recommended_books": recommendations})

def find_closest_book(input):
    book_titles = df['title'].to_list()
    closest_matches = process.extract(input, book_titles, limit= 10)
    # print(closest_matches)
    sorted_matches = sorted(closest_matches, key=lambda x: x[1], reverse=True)
    # print(sorted_matches)
    return sorted_matches[0][0]


recommendations = recommend_books(find_closest_book("Harry Potter"))
print("Recommended books:", recommendations)    

if __name__ == '__main__':
    app.run(debug=True)