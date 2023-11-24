import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import operator

def read_data(filepath='ml-1m/ratings.dat'):
    data = pd.io.parsers.read_csv(filepath, 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')

    return data

def get_user_item_matrix(data):
    matrix = data.pivot(index='movie_id', columns='user_id', values='rating')
    # normalize user-item matrix by each users mean
    matrix_norm = matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

    return matrix_norm


def get_similarity_matrix(matrix_norm):
    
    # calculate cosine similarity matrix
    item_similarity_matrix = cosine_similarity(matrix_norm.fillna(0))
    item_similarity_matrix_matrix = pd.DataFrame(item_similarity_matrix, index=matrix_norm.index, columns=matrix_norm.index)

    return item_similarity_matrix_matrix


def watched_movies(matrix_norm, picked_userid):
    '''
    Output: a dataframe of movies that the target user has watched,
         two columns: movie_id, rating
            sorted by rating in descending order
    '''

    # Movies that the target user has watched
    picked_userid_watched = pd.DataFrame(matrix_norm.loc[:, picked_userid].dropna(axis=0, how='all')\
                            .sort_values(ascending=False))\
                            .rename(columns={picked_userid:'rating'})
    
    return picked_userid_watched


def unwatched_movies(matrix_norm, picked_userid):
    '''
    Output: list, movie IDs that the target user has not watched
    '''

    # Movies that the target user has not watched
    picked_userid_unwatched = pd.DataFrame(matrix_norm.loc[:, picked_userid].isna())
    # a list of movie IDs that the target user has not watched
    picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[picked_userid]==True].index.tolist()
    return picked_userid_unwatched


def get_predicted_rating(picked_movie_id, picked_userid_watched, item_similarity_matrix, number_of_similar_items=5):
    '''
    Output: predicted rating for one given movie
    '''
    picked_movie_similarity_score = item_similarity_matrix[[picked_movie_id]].rename(columns={'index': 'movie_id', picked_movie_id:'similarity_score'})
    picked_userid_watched_similarity = pd.merge(left=picked_userid_watched, 
                                                right=picked_movie_similarity_score, 
                                                on='movie_id', 
                                                how='inner').sort_values('similarity_score', ascending=False)[:number_of_similar_items]
    # return picked_userid_watched_similarity
    try:
        predicted_rating = round(np.average(picked_userid_watched_similarity['rating'], 
                                        weights=picked_userid_watched_similarity['similarity_score']), 6)

    except ZeroDivisionError:
        print(f'No similar items found for the target movie {picked_movie_id}')
        predicted_rating = 0
    
    return predicted_rating


def predict_ratings(picked_userid, movie_id, item_similarity_matrix, matrix_norm):
    '''
    Input:
        picked_userid: int
        movie_id: int
        item_similarity_matrix: dataframe
        matrix_norm: dataframe
    Output:
        predicted_rating: float
    '''
    # Movies that the target user has watched
    picked_userid_watched = watched_movies(matrix_norm, picked_userid)
    # Predicted rating for the target movie
    predicted_rating = get_predicted_rating(movie_id, picked_userid_watched, item_similarity_matrix)
    
    return predicted_rating


def predict_one_user(picked_userid, item_similarity_matrix, matrix_norm, mv_pct=0.05):

    # Movies that the target user has watched
    picked_userid_watched = watched_movies(matrix_norm, picked_userid)
    # print(picked_userid_watched.shape[0])

    # randomly pick 10% movies
    if round(picked_userid_watched.shape[0]*mv_pct) > 0:
        n = round(picked_userid_watched.shape[0]*mv_pct)
    else:
        n = 1
    
    random_movieids = np.random.choice(picked_userid_watched.index, n, replace=False)
    
    predictions = []
    for picked_movie_id in random_movieids:
        # get the predicted rating for each user-movie pair
        predicted_rating = predict_ratings(picked_userid, picked_movie_id, item_similarity_matrix, matrix_norm)
        # append the predicted rating to the dataframe
        predictions.append({'user_id': picked_userid, 'movie_id': picked_movie_id, 'predicted_rating': predicted_rating})
        
    return predictions


def generate_predicted_ratings(matrix_norm, item_similarity_matrix, user_pct=0.05):
    '''
    Output:
        MAE: float
        RMSE: float
    '''

    # randomly pick 10% users
    picked_userids = np.random.choice(matrix_norm.columns, size=round(matrix_norm.shape[1]*user_pct), replace=False)

    predictions_all_user = []
    # loop through the picked users and movies
    for picked_userid in picked_userids:
        # print(picked_userid)
        # list of dictionaries
        predictions = predict_one_user(picked_userid, item_similarity_matrix, matrix_norm)
        predictions_all_user.extend(predictions)

    predicted_ratings_df = pd.DataFrame(predictions_all_user)
    return predicted_ratings_df


def calculate_MAE_RMSE(predicted_ratings_df, data):

    # merge the predicted ratings with the original ratings
    predicted_ratings = pd.merge(left=predicted_ratings_df, 
                                right=data[['user_id', 'movie_id', 'rating']], 
                                on=['user_id', 'movie_id'], 
                                how='inner')

    # calculate MAE and RMSE
    MAE = np.mean(abs(predicted_ratings['predicted_rating'] - predicted_ratings['rating']))
    RMSE = np.sqrt(np.mean((predicted_ratings['predicted_rating'] - predicted_ratings['rating'])**2))

    return MAE, RMSE


def recommend_items(picked_userid, item_similarity_matrix, matrix_norm, top_n=10):
    # Movies that the target user has watched
    picked_userid_watched = watched_movies(matrix_norm, picked_userid)
    
    # Movies that the target user has not watched
    picked_userid_unwatched = unwatched_movies(matrix_norm, picked_userid)

    rating_prediction = {}

    # loop through unwatched movies
    for movie_id in picked_userid_unwatched:
        # print(movie_id)
        predicted_rating = get_predicted_rating(movie_id, picked_userid_watched, item_similarity_matrix)
        # Save the predicted rating in the dictionary
        rating_prediction[movie_id] = predicted_rating

    return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:top_n]


def recommend_top_n_to_user(picked_userid, top_n=10):
    '''
    Input:
        picked_userid: int
        top_n: int, number of recommendations
    '''

    data = read_data()

    # if the file exists, read the file
    try:
        matrix_norm = pd.read_csv('matrix_norm.csv', index_col=0)
        item_similarity_matrix = pd.read_csv('item_similarity_matrix.csv', index_col=0)
        print('Read the saved matrices from the local directory')
    # if the file does not exist, calculate the matrices
    except FileNotFoundError:
        matrix_norm = get_user_item_matrix(data)
        # save the normalized user-item matrix
        matrix_norm.to_csv('matrix_norm.csv')
        item_similarity_matrix = get_similarity_matrix(matrix_norm)
        # save the item similarity matrix
        item_similarity_matrix.to_csv('item_similarity_matrix.csv')

    recommendations = recommend_items(picked_userid, item_similarity_matrix, matrix_norm, top_n=top_n)
    return recommendations