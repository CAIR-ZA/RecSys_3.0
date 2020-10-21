#Baseline(Popularity)  and Memory-Based Models code

import numpy as np
import pandas as pd

data = pd.read_csv('ratings.csv')
data.head(10)

movie_titles = pd.read_csv("movies.csv")
movie_titles.head(10)

data=data.merge(movie_titles,on='movieId',how="left")
data.head(10)

Average_ratings=pd.DataFrame(data.groupby('title')['rating'].mean())
Average_ratings.head(10)

Average_ratings["Total_Rating"]=pd.DataFrame(data.groupby('title')['rating'].count())
Average_ratings.tail(10)

#Compute the Histogram.
import matplotlib.pyplot as plt
# %matplotlib inline
Average_ratings['rating'].hist(bins=50)

Average_ratings["Total_Rating"].hist(bins=60)
plt.xlabel('Number of ratings per movie')
plt.ylabel('Count of ratings')
plt.title('Histogram of total ratings')
plt.grid(True)
plt.show()

#Scatter plot 
import seaborn as sns
h=sns.jointplot(x='rating', y='Total_Rating', data=Average_ratings, kind="reg")
h.set_axis_labels('Rating Score', 'Number of ratings')
h.fig.suptitle("Scatter plot of the total number of rating per interval.")
h.fig.tight_layout()
h.fig.subplots_adjust(top=0.95)

movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
movie_matrix.head()

Average_ratings.sort_values('Total_Rating', ascending=False).head(10)

FG_user_rating=movie_matrix['Forrest Gump (1994)']
#ShawRed_user_rating=movie_matrix['Shawshank Redemption, The (1994)']

correlations = movie_matrix.corrwith(FG_user_rating)
correlations.head(10)

recommendation = pd.DataFrame(correlations,columns=['Correlation'])
recommendation.dropna(inplace=True)
recommendation = recommendation.join(Average_ratings['Total_Rating'])

recc = recommendation[recommendation['Total_Rating']>100].sort_values('Correlation',ascending=False).reset_index()

recc = recc.merge(movie_titles,on='title', how='left')
recc.head(10)

import seaborn as sns
sns.jointplot(x='rating', y='Total_Rating', data=Average_ratings)

movie_matrix = movies.pivot_table(index='userId', columns='title', values='rating').fillna(0)
movie_matrix.head()

df_movie_features = movies.pivot(
    index='movieId',
    columns='userId',
    values='rating'
).fillna(0)
mat_movie_features = csr_matrix(df_movie_features.values)

df_movie_features.head()

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
num_users = len(data.userId.unique())
num_items = len(data.movieId.unique())
#No. of ratings per movie:
df_movies = pd.DataFrame(data.groupby('movieId').size(), columns=['count'])
df_movies.head()

popularity_thres = 50
popular_movies = list(set(df_movies.query('count >= @popularity_thres').index))
df_drop_movies = data[data.movieId.isin(popular_movies)]
print('original ratings data: ', data.shape)
print('ratings data after dropping unpopular movies: ', df_drop_movies.shape)
#No. of ratings per given user = pd.DataFrame(df_drop_movies.groupby('userId').size(), columns=['count'])
df_user.head()

ratings_thres = 50 #k=50 threshold.
active_users = list(set(df_user.query('count >= @ratings_thres').index))
df_drop_users = df_drop_movies[df_drop_movies.userId.isin(active_users)]
movie_user_mat = df_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)

