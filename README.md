# A Simple Recommender System: Item-Based Collaborative Filtering with the MovieLens Dataset

**Item-based collaborative filtering** is a type of recommendation system that uses the similarity between items to make recommendations. It is based on the idea that if two items are similar, then a user who likes one item is likely to like the other item as well.

The code provided here is an implementation of item-based collaborative filtering using the [MovieLens dataset (1M)](https://grouplens.org/datasets/movielens/1m/). This dataset contains 1 million ratings from 6000 users on 4000 movies.

To run the code, you need to download the dataset from [here](https://grouplens.org/datasets/movielens/1m/) and place the unzipped folder (named `ml-1m`) in the same directory as the code.

The `item_based_utils.py` file contains the implementation of the item-based collaborative filtering algorithm.

The `item_based.ipynb` notebook demonstrates how to utilize the code to **make recommendations for a user** and **evaluate the accuracy** of the recommender system.
