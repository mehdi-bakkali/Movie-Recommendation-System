# Movie recommendation system in R

This project adapts [DataFlair's R movie recommendation tutorial](https://data-flair.training/blogs/data-science-r-movie-recommendation-system/) into a reproducible, memory-conscious workflow for a larger MovieLens-format dataset. It explores the ratings and genres, builds an item based collaborative filtering model, recommends unseen movies, and compares recommendation quality with a popularity baseline.

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

Keep the supplied file names and columns. The Kaggle dataset is much larger than the tutorial's sample, so the scripts build a sparse ratings matrix, require at least 50 training ratings per movie, and limit model fitting to 10,000 users and the 3,000 most-rated movies. This bounds the memory used by recommenderlab's internal calculations. The recommendation run keeps the most active users and always includes a requested user when their ID exists in the ratings. The evaluation run selects a seeded sample of up to 10,000 eligible users. The underlying MovieLens data format uses `userId`, `movieId`, `rating`, and `timestamp` columns in `ratings.csv`; the timestamp is not needed by this project. [MovieLens file format](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)

Do not commit the large raw CSV files to a public portfolio repository. Include the Kaggle source link and follow the license terms shown on its dataset page.

## Step 3: Explore and prepare the data

Run the analysis script first:

```sh
Rscript analyze.R
```

It creates a dataset summary, rating distribution, most-rated movie table, multi-hot movie genre table, per-user favorite counts, and a centered-rating sample. It also binarizes sampled ratings into favorite (4 or 5) and not-favorite values, and saves rating, normalized-rating, and cosine-similarity charts for a small sample of active users and popular movies. Movies with multiple genres have a 1 in each matching genre column. Ratings are centered by each user's mean to show differences from their typical score. Ratings of 4 or 5 are counted as favorites, matching the evaluation threshold.

Analysis files and charts are written under `outputs/`, which is ignored by Git. The script limits heatmaps and similarity diagnostics to small samples, and caps the normalization sample at 500,000 ratings so the large source files do not require a dense full-catalog matrix.

## Step 4: Run the recommender

From the project folder:

```sh
Rscript recommend.R
```

That uses the first user with enough ratings. To choose a user and recommendation count:

```sh
Rscript recommend.R 1 10
```

The program prints recommendations and saves a CSV in `outputs/`. If user 1 is not present after filtering, choose a `userId` from `data/ratings.csv` that has rated several movies.

## Step 5: Evaluate recommendation quality

Run a reproducible comparison of item based collaborative filtering (IBCF) against a popularity baseline:

```sh
Rscript evaluate.R
```

The default run uses seed `20261025` and evaluates the top 10. It samples up to 10,000 eligible users. For each user with more than five ratings, it holds out 20% of that user's ratings while keeping at least five ratings as the known profile. Held-out ratings of 4 or 5 count as favorites. Movies already rated by that user are excluded from recommendations. The popularity baseline ranks movies by their number of training ratings. Movies with fewer than 50 training ratings are excluded, and the evaluation catalog is limited to the 3,000 most-rated movies.

The summary uses macro averages, so each evaluated user contributes equally. `Precision@10` is hits divided by 10. `Recall@10` is hits divided by the number of that user's held-out favorites that were available in the training catalog. Results are written to `outputs/evaluation_summary.csv` and `outputs/evaluation_by_user.csv`. You can change the seed and recommendation count, for example `Rscript evaluate.R 42 20`.

This is one random holdout split, so results can vary with the seed. For a stronger report, repeat the evaluation with several seeds or use cross-validation, then report the average and variability. Compare both methods on the same users and split.

## How the project works

1. `analyze.R` summarizes ratings and popularity, creates multi-hot genre features, centers a bounded ratings sample by user, and visualizes sampled user/item similarities.
2. `recommend.R` reads the movie metadata and ratings, then matches ratings to known movies.
3. It converts the ratings table into a sparse user-by-movie matrix and filters very sparse users and movies.
4. It fits an item based collaborative filtering (IBCF) model and recommends unrated movies for the selected user, translating movie IDs to titles.
5. `evaluate.R` uses a within-user holdout to compare IBCF against most popular movies on Precision@K and Recall@K.

## Generated analysis files

`analyze.R` writes `dataset_summary.csv`, `rating_distribution.csv`, `movie_popularity.csv`, `movie_genre_encoding.csv`, `genre_counts.csv`, `favorite_counts_by_user.csv`, `rating_preparation_summary.csv`, and `normalized_rating_sample.csv`, plus PNG charts and sampled similarity matrices. Recommendation and evaluation files are created by the other scripts. These generated outputs are reproducible and intentionally excluded from version control.


