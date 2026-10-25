# Movie recommendation system in R

This project follows DataFlair's item based collaborative filtering tutorial. It uses movie ratings to find movies with similar rating patterns, then recommends movies a selected user has not rated.

## Step 1: Install R and packages

Install R, then open a terminal in this project folder and install the required R packages:

```r
install.packages(c("recommenderlab", "data.table", "Matrix"), repos = "https://cloud.r-project.org")
```

`recommenderlab` may require a recent R version. If your system cannot install it, share the installation error and your operating system and we can adapt the project.

## Step 2: Add the data

Use the [27M Movie Ratings dataset on Kaggle](https://www.kaggle.com/datasets/pa13lito/27m-movie-ratings). Create a `data` folder and place its ratings and movie metadata files there. The scripts support either of these layouts:

- Simple MovieLens layout: `movies.csv` with `movieId,title` (and optionally `genres`), plus `ratings.csv` with `userId,movieId,rating`.
- Full Movies Dataset layout: `ratings.csv`, `movies_metadata.csv` with `id,title` (and optionally `genres`), and `links.csv` with `movieId,tmdbId`. The recommender joins the metadata through `links.csv`.

Keep the supplied file names and columns. The Kaggle dataset is much larger than the tutorial's sample, so the scripts build a sparse ratings matrix and only keep movies with at least 50 training ratings when fitting item similarities. The underlying MovieLens data format uses `userId`, `movieId`, `rating`, and `timestamp` columns in `ratings.csv`; the timestamp is not needed by this project. [MovieLens file format](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)

Do not commit the large raw CSV files to a public portfolio repository. Include the Kaggle source link and follow the license terms shown on its dataset page.

## Step 3: Run the recommender

From the project folder:

```sh
Rscript recommend.R
```

That uses the first user with enough ratings. To choose a user and recommendation count:

```sh
Rscript recommend.R 1 10
```

The program prints recommendations and saves a CSV in `outputs/`. If user 1 is not present after filtering, choose a `userId` from `data/ratings.csv` that has rated several movies.

## Evaluate recommendation quality

Run a reproducible comparison of item based collaborative filtering (IBCF) against a popularity baseline:

```sh
Rscript evaluate.R
```

The default run uses seed `20261025` and evaluates the top 10. For each user with more than five ratings, it holds out 20% of that user's ratings while keeping at least five ratings as the known profile. Held-out ratings of 4 or 5 count as favorites. Movies already rated by that user are excluded from recommendations. The popularity baseline ranks movies by their number of training ratings. Movies with fewer than 50 training ratings are excluded from the evaluation catalog to keep item similarity computation manageable on the larger data.

The summary uses macro averages, so each evaluated user contributes equally. `Precision@10` is hits divided by 10. `Recall@10` is hits divided by the number of that user's held-out favorites that were available in the training catalog. Results are written to `outputs/evaluation_summary.csv` and `outputs/evaluation_by_user.csv`. You can change the seed and recommendation count, for example `Rscript evaluate.R 42 20`.

This is one random holdout split, so results can vary with the seed. For a stronger report, repeat the evaluation with several seeds or use cross-validation, then report the average and variability. Compare both methods on the same users and split.

## How the project works

1. Read the movie metadata and ratings, then match ratings to known movies.
2. Convert the ratings table into a sparse user-by-movie matrix.
3. Filter very sparse users and movies so similarity estimates have useful data.
4. Fit an item based collaborative filtering (IBCF) model with cosine similarity.
5. Recommend unrated movies for the selected user and translate movie IDs to titles.


