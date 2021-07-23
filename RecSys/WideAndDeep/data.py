import pandas as pd
import numpy as np
import pickle

# ========================================
# Load MovieLens 1M datasets
# ========================================
data_movies = pd.read_csv('./ml-1m/movies.dat', sep = "::", names = ['movieid','title','genres'])
data_ratings = pd.read_csv('./ml-1m/ratings.dat', sep = "::", names = ['userid','movieid','ratings','timestamp'])
data_users = pd.read_csv('./ml-1m/users.dat', sep = "::", names = ['userid','gender','age','occupation','zipcode'])

# ========================================
# Wide Data => One-hot-encoding 형태로 변환 
# gender, age, genres 세 요소 사용
# cross-product transformation은 age와 gender 간 형성
# ========================================
# ========================================
# data_movies에서 genres의 unique를 
# 추출하여 one-hot-encoder 형태로 변환
# ex) Animation|Chidren's 인 경우 [1,1,0,0,0, ... ]
# ========================================
# genres's unique 추출
genres_unique = np.unique(data_movies.genres.str.split('|').apply(lambda x: pd.Series(x)).stack())

# 각각의 movie가 속한 장르를 one-hot-encoding으로 변환
movie_genres = data_movies.genres.str.split('|')
movie_genres_one_hot = pd.DataFrame(0, index=np.arange(len(data_movies)), columns = genres_unique)
for id, i in enumerate(movie_genres):
   for j in i:
      movie_genres_one_hot[j][id] = 1

# one-hot-encoding 표현으로 바꾸고, genres 열 삭제
data_movies = pd.concat([data_movies, movie_genres_one_hot], axis=1).drop(['genres'], axis = 1)

temp1 = pd.merge(data_ratings, data_users, how='outer', on = 'userid')
data_merge = pd.merge(data_movies,temp1, how = 'outer', on = 'movieid')
data_merge.dropna(inplace = True)

# rating이 1,2,3인 경우 비선호 0 / 4,5인 경우 선호 1
data_merge.ratings = data_merge.ratings.astype('int')
data_merge.ratings = np.where(data_merge.ratings < 3.5 , 0, 1)

data_wide = data_merge.drop(['title','timestamp', 'zipcode', 'occupation'], axis = 1)

# gender, age 범주형으로 변경
data_wide.gender = np.where(data_wide.gender == 'F','female','male')
data_wide.age = np.where(data_wide.age < 30,'youth','adult')

# one-hot-encoding 형태로 변경
gender_one_hot = pd.get_dummies(data_wide['gender'])
age_one_hot = pd.get_dummies(data_wide['age'])
data_wide = pd.concat([data_wide, gender_one_hot, age_one_hot], axis = 1).drop(['gender', 'age'], axis = 1)

# cross-product transformation data 추가
data_wide['female-adult']= data_wide.female * data_wide.adult
data_wide['female-youth']= data_wide.female * data_wide.youth
data_wide['male-adult'] = data_wide.male * data_wide.adult
data_wide['male-youth'] = data_wide.male * data_wide.youth

# 열 순서 변경
data_wide = data_wide[['userid','movieid', 'Action', 'Adventure', 'Animation', "Children's",
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
       'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
       'Western', 'female', 'male', 'adult', 'youth','female-adult', 'female-youth', 'male-adult', 'male-youth', 'ratings']]

with open("data_wide.pkl", "wb") as f:
    pickle.dump(data_wide, f)

print("")
print("###################################")
print("###### data_wide.pkl - Saved ######")
print("###################################")

# ========================================
# Deep Data 
# gender, age, genres, occupation 사용
# genres는 one-hot-encoder로 사용하고, gender, age, occupation은 그냥 그대로 (범주형)사용
# ========================================
data_deep = data_merge.copy()
data_deep.drop(['timestamp','zipcode','title'], axis = 1, inplace = True)

# 열 순서 변경
data_deep = data_deep[['userid','movieid', 'Action', 'Adventure', 'Animation', "Children's",
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
       'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
       'Western',  'gender', 'age', 'occupation','ratings']]

# ========================================
# 범주형 데이터를 0부터 시작하는 정수형 카테고리로 변환
# ========================================
data_deep.age = data_deep.age.astype('category').cat.codes
data_deep.gender = data_deep.gender.astype('category').cat.codes
data_deep.occupation = data_deep.occupation.astype('category').cat.codes
data_deep.userid = data_deep.userid.astype('category').cat.codes
data_deep.movieid = data_deep.movieid.astype('category').cat.codes

with open("data_deep.pkl", "wb") as f:
    pickle.dump(data_deep, f)

print("")
print("###################################")
print("###### data_deep.pkl - Saved ######")
print("###################################")

