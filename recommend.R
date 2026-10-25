
required_packages <- c("recommenderlab", "data.table", "Matrix")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install the required packages first: install.packages(c(",
       paste(sprintf('"%s"', missing_packages), collapse = ", "), "))", call. = FALSE)
}

suppressPackageStartupMessages({
  library(recommenderlab)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
requested_user <- if (length(args) >= 1) args[[1]] else NULL
recommendation_count <- if (length(args) >= 2) as.integer(args[[2]]) else 10L
if (is.na(recommendation_count) || recommendation_count < 1L) {
  stop("The number of recommendations must be a positive integer.", call. = FALSE)
}

movies_path <- file.path("data", "movies.csv")
metadata_path <- file.path("data", "movies_metadata.csv")
ratings_path <- file.path("data", "ratings.csv")
if ((!file.exists(movies_path) && !file.exists(metadata_path)) || !file.exists(ratings_path)) {
  stop("Dataset files not found. Put ratings.csv and either movies.csv or movies_metadata.csv in data/; see README.md.", call. = FALSE)
}

ratings <- fread(ratings_path)
required_rating_columns <- c("userId", "movieId", "rating")
if (!all(required_rating_columns %in% names(ratings))) {
  stop("Expected ratings.csv columns userId,movieId,rating.", call. = FALSE)
}

if (file.exists(movies_path)) {
  movies <- fread(movies_path, encoding = "UTF-8")
  if (!all(c("movieId", "title") %in% names(movies))) {
    stop("movies.csv must include movieId and title columns.", call. = FALSE)
  }
} else {
  links_path <- file.path("data", "links.csv")
  if (!file.exists(links_path)) {
    stop("For movies_metadata.csv, links.csv is also required to map MovieLens movieId to metadata id.", call. = FALSE)
  }
  metadata <- fread(metadata_path, encoding = "UTF-8")
  links <- fread(links_path)
  if (!all(c("id", "title") %in% names(metadata)) || !all(c("movieId", "tmdbId") %in% names(links))) {
    stop("Expected movies_metadata.csv columns id,title and links.csv columns movieId,tmdbId.", call. = FALSE)
  }
  metadata[, tmdb_key := as.character(id)]
  links[, tmdb_key := as.character(suppressWarnings(as.integer(as.numeric(tmdbId))))]
  metadata_columns <- c("tmdb_key", "title", intersect("genres", names(metadata)))
  movies <- merge(links[!is.na(tmdb_key), .(movieId, tmdb_key)],
                  metadata[, ..metadata_columns], by = "tmdb_key")
  if (!"genres" %in% names(movies)) movies[, genres := NA_character_]
  movies <- movies[, .(movieId, title, genres)]
}

# Keep only ratings for movies with metadata and build user-by-movie ratings.
ratings <- ratings[movieId %in% movies$movieId, .(userId, movieId, rating)]
ratings <- ratings[!is.na(userId) & !is.na(movieId) & !is.na(rating)]
if (nrow(ratings) == 0L) stop("No usable ratings remain after matching movies.", call. = FALSE)
ratings <- ratings[, .(rating = mean(as.numeric(rating))), by = .(userId, movieId)]

user_ids <- sort(unique(as.character(ratings$userId)))
movie_ids <- sort(unique(as.character(ratings$movieId)))
sparse_values <- Matrix::sparseMatrix(
  i = match(as.character(ratings$userId), user_ids),
  j = match(as.character(ratings$movieId), movie_ids),
  x = as.numeric(ratings$rating),
  dims = c(length(user_ids), length(movie_ids)),
  dimnames = list(user_ids, movie_ids)
)
rating_matrix <- new("realRatingMatrix", data = sparse_values)

# Filter sparse users and rare movies before fitting item similarities.
min_user_ratings <- min(5L, max(1L, floor(ncol(rating_matrix) * 0.01)))
min_movie_ratings <- 50L
max_model_items <- 5000L
movie_counts <- colCounts(rating_matrix)
eligible_movies <- which(movie_counts >= min_movie_ratings)
if (length(eligible_movies) > max_model_items) {
  keep_order <- order(-movie_counts[eligible_movies], colnames(rating_matrix)[eligible_movies])
  eligible_movies <- eligible_movies[keep_order[seq_len(max_model_items)]]
}
rating_matrix <- rating_matrix[, eligible_movies]
rating_matrix <- rating_matrix[rowCounts(rating_matrix) >= min_user_ratings, ]
if (nrow(rating_matrix) < 2L || ncol(rating_matrix) < 2L) {
  stop("Not enough ratings after filtering. Try a larger MovieLens dataset.", call. = FALSE)
}

if (is.null(requested_user)) requested_user <- rownames(rating_matrix)[[1]]
user_position <- match(as.character(requested_user), rownames(rating_matrix))
if (is.na(user_position)) {
  stop("User not found after filtering. Choose a userId with enough ratings; see data/ratings.csv.", call. = FALSE)
}

model <- Recommender(rating_matrix, method = "IBCF", parameter = list(k = min(30L, ncol(rating_matrix) - 1L)))
prediction <- predict(model, rating_matrix[user_position, ], n = min(recommendation_count, ncol(rating_matrix)))
recommended_ids <- prediction@itemLabels[prediction@items[[1]]]

result_columns <- c("movieId", "title", intersect("genres", names(movies)))
result <- movies[match(as.integer(recommended_ids), movies$movieId), ..result_columns]
result <- result[!is.na(movieId)]
if (nrow(result) == 0L) {
  message("No unseen recommendations were available for this user. Try another userId.")
} else {
  cat(sprintf("Top recommendations for user %s:\n", requested_user))
  print(result)
  dir.create("outputs", showWarnings = FALSE)
  fwrite(result, file.path("outputs", paste0("recommendations-user-", requested_user, ".csv")))
  message("Saved recommendations under outputs/.")
}
