import pandas as pd
import numpy as np

# Load data
ratings_data = pd.read_csv('ratings.csv')
music_data = pd.read_csv('music.csv')

# Merge dataframes
merged_data = pd.merge(ratings_data, music_data, on='music_id')

# Calculate mean ratings for each music
mean_ratings = merged_data.groupby('title')['rating'].mean().reset_index()

# Create pivot table of ratings
ratings_matrix = merged_data.pivot_table(index='user_id', columns='title', values='rating')

# Fill missing values with 0
ratings_matrix.fillna(0, inplace=True)

# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
music_similarity = cosine_similarity(ratings_matrix.T)

# Create function to recommend music
def recommend_music(music_name):
    music_index = mean_ratings[mean_ratings['title'] == music_name].index[0]
    similar_music = music_similarity[music_index]
    recommended_music = list(mean_ratings.iloc[np.argsort(similar_music)]['title'].values)[1:6]
    return recommended_music

print(recommend_music("Stairway to Heaven"))

from flask import Flask, render_template, request

app = Flask(__name__)

def my_function(text):
    # do something with the text parameter and return an array
    return [text.upper(), text.lower(), len(text)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_text():
    user_text = request.form['user_text']
    result_array = recommend_music(user_text)
    return render_template('index.html', result_array=result_array)

if __name__ == '__main__':
    app.run()