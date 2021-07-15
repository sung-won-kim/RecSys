import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings("ignore")

# ========================================
# Input 검색할 영화 Title
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--userid', required = False, type = int, default= 0,
                    help='userid number (default : 0')
parser.add_argument('--recNum', required = False, type = int, default= 10,
help='Number of recommendations (default : 10')
args = parser.parse_args()

# ========================================
# Load Dataset
# ========================================
# load ratings.csv, pre_tmdb_5000_movies.csv from 'https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata.csv' (pre_tmdb_5000_movies : read ContentBasedFiltering.py)
def data_load(dataset):
   data = pd.read_csv(dataset)
   return data

# ========================================
# Preprocessing dataset
# ========================================
rating_data = data_load('./data/ratings_small.csv')
movie_data = data_load('./data/pre_tmdb_5000_movies.csv')
# 필요 없는 열 전처리  
rating_data.drop('timestamp',axis = 1, inplace = True)
# movieid column을 기준으로 dataset merge
movie_data.rename(columns = {'id': 'movieId'}, inplace = True)
user_movie_rating = pd.merge(rating_data, movie_data, on = 'movieId')
# user-item table을 rating을 속성으로 생성 (row가 user, col이 item)
data = user_movie_rating.pivot_table('rating', index = 'userId', columns = 'movieId').fillna(0)

# ========================================
# Bias
# ========================================
# 1. pivot table을 matrix로 변환
# 2. np.mean(axis=1)로 하나의 영화에 대한 각 사용자들이 매기는 평점 평균을 구함
# 3. 1에서 구한 값과 2에서 구한 값의 차를 구해 사용자-평균 데이터 값으로 변경
matrix = data.to_numpy() # pivot_table을 matrix로 변환
user_ratings_mean = np.mean(matrix,axis = 1) # 각 영화의 평균평점이 구해짐
matrix_user_mean = matrix - user_ratings_mean.reshape(-1,1) # R_user_mean : 사용자-영화에 대해 평균평점을 뺌

# ========================================
# Matrix Factorization with SVD
# ========================================
# scipy에서 제공해주는 svd.  
# U 행렬, sigma 행렬, V 전치 행렬을 반환.
U, sigma, Vt = svds(matrix_user_mean, k = 12)
sigma = np.diag(sigma) # 시그마를 대각행렬로 바꿔줌

# ========================================
# 원본 행렬로 복원 (내적)
# ========================================
svd_user_predicted_ratings = np.dot(np.dot(U,sigma),Vt) + user_ratings_mean.reshape(-1,1)

df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = data.columns)

# ========================================
# 사용자에게 추천 함수
# ========================================
# 인자로 사용자의 id, 영화 정보 테이블, 평점 테이블 등 받음
# 사용자 id에 SVD로 나온 결과의 영화 평점이 가장 높은 데이터 순 정렬
# 사용자가 본 데이터 제외
# 사용자가 안 본 영화에서 평점 높은 것 추천
def recommend_movies(df_svd_preds, user_id, ori_movies_df, ori_ratings_df, num_recommendations=5):
    
    #현재는 index로 적용이 되어있으므로 user_id - 1을 해야함.
    user_row_number = user_id - 1 
    
    # 최종적으로 만든 pred_df에서 사용자 index에 따라 영화 데이터 정렬 -> 영화 평점이 높은 순으로 정렬 됌
    sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
    
    # 원본 평점 데이터에서 user id에 해당하는 데이터를 뽑아낸다. 
    user_data = ori_ratings_df[ori_ratings_df.userId == user_id]
    
    # 위에서 뽑은 user_data와 원본 영화 데이터를 합친다. 
    user_history = user_data.merge(ori_movies_df, on = 'movieId').sort_values(['rating'], ascending=False)
    
    # 원본 영화 데이터에서 사용자가 본 영화 데이터를 제외한 데이터를 추출
    recommendations = ori_movies_df[~ori_movies_df['movieId'].isin(user_history['movieId'])]
    # 사용자의 영화 평점이 높은 순으로 정렬된 데이터와 위 recommendations을 합친다. 
    recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = 'movieId')
    # 컬럼 이름 바꾸고 정렬해서 return
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
                      

    return user_history, recommendations

already_rated, predictions = recommend_movies(df_svd_preds, args.userid, movie_data, rating_data, args.recNum)

print('already_rated')
print(already_rated.head(10))
print("here's recommendation for you")
print(predictions.head(10))