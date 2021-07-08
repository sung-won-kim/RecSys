import pandas as pd 
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# ========================================
# Input 검색할 영화 Title
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--title', required = False, default= 'The Dark Knight Rises',
                    help='검색할 영화 제목 (default : The Dark Knight Rises')
args = parser.parse_args()

# ========================================
# Loading Dataset
# ========================================
# load movies_metadata.csv (https://www.kaggle.com/rounakbanik/the-movies-dataset)
def data_load(dataset):
   data = pd.read_csv(dataset)
   data = data[['id','genres', 'vote_average', 'vote_count','popularity','title',  'tagline', 'overview']]
   data['vote_count'].fillna(0).astype(int)
   return data

# ========================================
# 전처리
# ========================================
   # weight rating (WR) = (v / (v+m)) x R + (m / (v+m)) * C
   # R : 개별 영화 평점, v : 개별 영화에 평점을 투표한 횟수, m : N위 안에 들기 위해 필요한 최소 투표 횟수, C : 전체 영화에 대한 평균 평점
   # vote 수가 적은 평점에 대한 불공정을 처리하는 방법
def weight_rating(data):
   # m을 찾기 위해 상위 90%라고 설정하였을 때 필요한 최소 vote_count 계산
   m = data['vote_count'].quantile(0.9)
   data90 = data.loc[data['vote_count'] >= m]
   C = data90['vote_average'].mean()
   v = data['vote_count']
   R = data['vote_average']
   return (v / (v+m) * R) + (m / (m + v) * C)

def score(data):
   data['score'] = weight_rating(data)
   return data

def preDict(data):
   # genres에서 속성값만 추출하여 전처리
   data['genres'] = data['genres'].apply(literal_eval)
   data['genres'] = data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))
   return data
   
def save(data):
   data.to_csv('./data/pre_tmdb_5000_movies.csv', index = False)

# ========================================
# 콘텐츠 기반 필터링 추천 (Content Based Filtering)
# ========================================
# 문자열을 숫자로 벡터화
def wordFit(data, col):
   count_vector = CountVectorizer(ngram_range=(1,3))
   c_vector_col = count_vector.fit_transform(data[col])
   return c_vector_col

# Cosine Similarity 
def cosSim(data, col):
   c_sim = cosine_similarity(wordFit(data,col), wordFit(data,col)).argsort()[:, ::-1]   
   return c_sim

def get_recommend_movie_list(data, movie_title, col = 'genres', top=30):
    # 특정 영화와 비슷한 영화를 추천해야 하기 때문에 '특정 영화' 정보를 뽑아낸다.
    target_movie_index = data[data['title'] == movie_title].index.values

    c_sim = cosSim(data, col)

    #코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
    sim_index = c_sim[target_movie_index, :top].reshape(-1)
    #본인을 제외
    sim_index = sim_index[sim_index != target_movie_index]

    #data frame으로 만들고 vote_count으로 정렬한 뒤 return
    result = data.iloc[sim_index].sort_values('score', ascending=False)[:10]
    return result

data = data_load('./data/movies_metadata.csv')
# print(score(data).head())
data = score(data)
data = preDict(data)
print(get_recommend_movie_list(data, movie_title = args.title))

