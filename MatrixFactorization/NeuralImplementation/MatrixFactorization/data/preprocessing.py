import pandas as pd

# ========================================
# Preprocessing dataset
# ========================================
# download dataset : https://www.kaggle.com/rounakbanik/the-movies-dataset?select=ratings.csv
rating_data = pd.read_csv('./data/ratings.csv')
movie_data = pd.read_csv('./data/movies_metadata.csv')

# movieid column을 기준으로 dataset merge
movie_data.rename(columns = {'id': 'movieId'}, inplace = True)
user_movie_rating = pd.merge(rating_data, movie_data, on = 'movieId')
# user-item table을 rating을 속성으로 생성 (row가 user, col이 item)
data = user_movie_rating.pivot_table('rating', index = 'userId', columns = 'movieId').fillna(0)
# 전처리한 file 저장
data.to_csv('./data/user_10466_movies_5084.csv', index = False)