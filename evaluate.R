# Compare item-based collaborative filtering with a most-rated-movies baseline.
# Run from the project folder with: Rscript evaluate.R [seed] [k]

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
seed <- if (length(args) >= 1) as.integer(args[[1]]) else 20261025L
k <- if (length(args) >= 2) as.integer(args[[2]]) else 10L
if (is.na(seed) || is.na(k) || k < 1L) {
  stop("Usage: Rscript evaluate.R [seed] [positive recommendation count]", call. = FALSE)
}
set.seed(seed)

ratings_path <- file.path("data", "ratings.csv")
if (!file.exists(ratings_path)) {
  stop("ratings.csv not found. Put it in the data/ folder; see README.md.", call. = FALSE)
}
ratings <- fread(ratings_path)
required_columns <- c("userId", "movieId", "rating")
if (!all(required_columns %in% names(ratings))) {
  stop("Expected ratings.csv columns userId,movieId,rating.", call. = FALSE)
}
ratings <- ratings[!is.na(userId) & !is.na(movieId) & !is.na(rating),
                   .(rating = mean(as.numeric(rating))), by = .(userId, movieId)]
if (nrow(ratings) == 0L) stop("No usable ratings found.", call. = FALSE)

# Hold out 20% of each eligible user's ratings while retaining a known profile.
holdout_fraction <- 0.20
minimum_known_ratings <- 5L
ratings[, n_user_ratings := .N, by = userId]
ratings[, holdout_count := if (.N > minimum_known_ratings) {
  min(max(1L, floor(.N * holdout_fraction)), .N - minimum_known_ratings)
} else 0L, by = userId]
ratings[, random_rank := frank(runif(.N), ties.method = "first"), by = userId]
test_ratings <- ratings[random_rank <= holdout_count, .(userId, movieId, rating)]
train_ratings <- ratings[random_rank > holdout_count, .(userId, movieId, rating)]
if (nrow(test_ratings) == 0L) {
  stop("No users have more than five ratings; use a larger dataset.", call. = FALSE)
}

user_ids <- sort(unique(as.character(train_ratings$userId)))
movie_ids <- sort(unique(as.character(train_ratings$movieId)))
sparse_values <- Matrix::sparseMatrix(
  i = match(as.character(train_ratings$userId), user_ids),
  j = match(as.character(train_ratings$movieId), movie_ids),
  x = as.numeric(train_ratings$rating),
  dims = c(length(user_ids), length(movie_ids)),
  dimnames = list(user_ids, movie_ids)
)
train_matrix <- new("realRatingMatrix", data = sparse_values)

# Remove rare items before building the item-item similarity model.
min_movie_ratings <- 50L
max_model_items <- 5000L
movie_counts <- colCounts(train_matrix)
eligible_movies <- which(movie_counts >= min_movie_ratings)
if (length(eligible_movies) > max_model_items) {
  keep_order <- order(-movie_counts[eligible_movies], colnames(train_matrix)[eligible_movies])
  eligible_movies <- eligible_movies[keep_order[seq_len(max_model_items)]]
}
train_matrix <- train_matrix[, eligible_movies]
train_matrix <- train_matrix[rowCounts(train_matrix) >= minimum_known_ratings, ]
if (nrow(train_matrix) < 2L || ncol(train_matrix) < 2L) {
  stop("Not enough user/item overlap for evaluation after the split.", call. = FALSE)
}

item_ids <- colnames(train_matrix)
item_popularity <- train_ratings[movieId %in% as.numeric(item_ids),
                                 .(rating_count = .N), by = movieId]
setorder(item_popularity, -rating_count, movieId)
popular_order <- as.character(item_popularity$movieId)

model <- Recommender(
  train_matrix,
  method = "IBCF",
  parameter = list(k = min(30L, ncol(train_matrix) - 1L))
)

eligible_test <- test_ratings[as.character(userId) %in% rownames(train_matrix) &
                                as.character(movieId) %in% item_ids]
relevant_by_user <- split(
  as.character(eligible_test[rating >= 4, movieId]),
  as.character(eligible_test[rating >= 4, userId])
)
users_to_score <- intersect(names(relevant_by_user), rownames(train_matrix))
if (length(users_to_score) == 0L) {
  stop("No held-out ratings of 4 or 5 remain for evaluation. Check the ratings scale and dataset.", call. = FALSE)
}

score_rows <- vector("list", length(users_to_score) * 2L)
out_index <- 1L
for (user_id in users_to_score) {
  user_position <- match(user_id, rownames(train_matrix))
  known_ids <- as.character(train_ratings[
    as.character(userId) == user_id & as.character(movieId) %in% item_ids,
    movieId
  ])
  relevant_ids <- unique(relevant_by_user[[user_id]])
  candidates <- setdiff(item_ids, known_ids)
  recommendations <- character()

  if (length(candidates) > 0L) {
    prediction <- tryCatch(
      predict(model, train_matrix[user_position, ], n = min(k, length(item_ids))),
      error = function(e) NULL
    )
    if (!is.null(prediction)) {
      recommendations <- prediction@itemLabels[prediction@items[[1]]]
      recommendations <- head(intersect(recommendations, candidates), k)
    }
  }
  popular_recommendations <- head(intersect(popular_order, candidates), k)

  for (entry in list(
    list(method = "IBCF", items = recommendations),
    list(method = "Most popular", items = popular_recommendations)
  )) {
    hits <- length(intersect(entry$items, relevant_ids))
    score_rows[[out_index]] <- data.table(
      userId = user_id,
      method = entry$method,
      relevant_held_out = length(relevant_ids),
      recommendations_returned = length(entry$items),
      hits = hits,
      precision_at_k = hits / k,
      recall_at_k = hits / length(relevant_ids)
    )
    out_index <- out_index + 1L
  }
}

by_user <- rbindlist(score_rows)
summary <- by_user[, .(
  users_evaluated = .N,
  mean_relevant_held_out = mean(relevant_held_out),
  mean_recommendations_returned = mean(recommendations_returned),
  precision_at_k = mean(precision_at_k),
  recall_at_k = mean(recall_at_k)
), by = method]
summary[, `:=`(k = k, favorite_threshold = 4, seed = seed)]
setcolorder(summary, c("method", "k", "favorite_threshold", "seed", "users_evaluated",
                       "mean_relevant_held_out", "mean_recommendations_returned",
                       "precision_at_k", "recall_at_k"))

dir.create("outputs", showWarnings = FALSE)
fwrite(summary, file.path("outputs", "evaluation_summary.csv"))
fwrite(by_user, file.path("outputs", "evaluation_by_user.csv"))
cat(sprintf("Evaluation: %d users; favorites are held-out ratings >= 4; Precision@%d uses a fixed denominator of %d.\n",
            length(users_to_score), k, k))
print(summary)
message("Saved aggregate and per-user results in outputs/.")
