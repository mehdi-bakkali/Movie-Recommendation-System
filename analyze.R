# Explore and prepare a MovieLens-style movie ratings dataset.
# Run from the project folder with: Rscript analyze.R

required_packages <- c("data.table", "Matrix", "recommenderlab")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install the required packages first: install.packages(c(",
       paste(sprintf('"%s"', missing_packages), collapse = ", "), "))", call. = FALSE)
}

suppressPackageStartupMessages({
  library(data.table)
  library(recommenderlab)
})

ratings_path <- file.path("data", "ratings.csv")
movies_path <- file.path("data", "movies.csv")
metadata_path <- file.path("data", "movies_metadata.csv")
if (!file.exists(ratings_path) || (!file.exists(movies_path) && !file.exists(metadata_path))) {
  stop("Dataset files not found. Put ratings.csv and either movies.csv or movies_metadata.csv in data/; see README.md.", call. = FALSE)
}

ratings <- fread(ratings_path)
if (!all(c("userId", "movieId", "rating") %in% names(ratings))) {
  stop("Expected ratings.csv columns userId,movieId,rating.", call. = FALSE)
}
ratings <- ratings[!is.na(userId) & !is.na(movieId) & !is.na(rating),
                   .(rating = mean(as.numeric(rating))), by = .(userId, movieId)]
if (nrow(ratings) == 0L) stop("No usable ratings found.", call. = FALSE)

if (file.exists(movies_path)) {
  movies <- fread(movies_path, encoding = "UTF-8")
  if (!all(c("movieId", "title") %in% names(movies))) {
    stop("movies.csv must include movieId and title columns.", call. = FALSE)
  }
  if (!"genres" %in% names(movies)) movies[, genres := NA_character_]
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
movies <- unique(movies, by = "movieId")
ratings <- ratings[movieId %in% movies$movieId]
if (nrow(ratings) == 0L) stop("No ratings match the movie metadata.", call. = FALSE)

output_dir <- "outputs"
dir.create(output_dir, showWarnings = FALSE)
set.seed(42)

# Core descriptive statistics and rating distribution.
rating_summary <- ratings[, .(
  ratings = .N,
  users = uniqueN(userId),
  movies = uniqueN(movieId),
  mean_rating = mean(rating),
  median_rating = median(rating),
  rating_sd = sd(rating),
  sparsity = 1 - .N / (uniqueN(userId) * uniqueN(movieId))
)]
fwrite(rating_summary, file.path(output_dir, "dataset_summary.csv"))
rating_distribution <- ratings[, .(rating_count = .N), by = rating][order(rating)]
fwrite(rating_distribution, file.path(output_dir, "rating_distribution.csv"))
png(file.path(output_dir, "rating_distribution.png"), width = 1000, height = 650)
barplot(rating_distribution$rating_count, names.arg = rating_distribution$rating,
        col = "steelblue", xlab = "Rating", ylab = "Number of ratings",
        main = "Movie rating distribution")
dev.off()

# Most viewed/rated movies and a readable popularity chart.
movie_popularity <- ratings[, .(
  rating_count = .N,
  average_rating = mean(rating)
), by = movieId]
movie_popularity <- merge(movie_popularity, movies[, .(movieId, title)], by = "movieId", all.x = TRUE)
setorder(movie_popularity, -rating_count, title)
fwrite(movie_popularity, file.path(output_dir, "movie_popularity.csv"))
top_movies <- head(movie_popularity, 10L)
png(file.path(output_dir, "top_movies.png"), width = 1300, height = 800)
par(mar = c(10, 5, 4, 2))
barplot(rev(top_movies$rating_count), names.arg = rev(top_movies$title), las = 2,
        col = "steelblue", ylab = "Number of ratings", main = "Top 10 most-rated movies")
dev.off()

# Multi-hot genre encoding: a movie can belong to several genres.
genre_rows <- lapply(seq_len(nrow(movies)), function(i) {
  value <- movies$genres[[i]]
  if (is.na(value) || !nzchar(value) || value == "(no genres listed)") return(NULL)
  data.table(movieId = movies$movieId[[i]], genre = strsplit(value, "\\|", fixed = FALSE)[[1]])
})
genre_long <- rbindlist(genre_rows, use.names = TRUE)
if (nrow(genre_long)) {
  genre_long <- unique(genre_long)
  genre_levels <- sort(unique(genre_long$genre))
  genre_encoding <- dcast(genre_long[, value := 1L], movieId ~ genre,
                          value.var = "value", fill = 0L)
  fwrite(genre_encoding, file.path(output_dir, "movie_genre_encoding.csv"))
  genre_counts <- genre_long[, .(movie_count = uniqueN(movieId)), by = genre][order(-movie_count)]
  fwrite(genre_counts, file.path(output_dir, "genre_counts.csv"))
  png(file.path(output_dir, "genre_counts.png"), width = 1200, height = 750)
  par(mar = c(10, 5, 4, 2))
  barplot(rev(genre_counts$movie_count), names.arg = rev(genre_counts$genre), las = 2,
          col = "darkseagreen", ylab = "Movies", main = "Movie counts by genre")
  dev.off()
} else {
  genre_levels <- character()
  message("No genre values found; skipped genre encoding and genre chart.")
}

# Normalize a reproducible sample by subtracting each user's mean rating.
# Keep this analysis bounded so it remains practical on a laptop.
normalization_sample_size <- min(nrow(ratings), 500000L)
normalization_sample <- ratings[sample.int(nrow(ratings), normalization_sample_size)]
normalization_sample[, user_mean := mean(rating), by = userId]
normalization_sample[, normalized_rating := rating - user_mean]
normalized_summary <- normalization_sample[, .(
  sampled_ratings = .N,
  mean_normalized_rating = mean(normalized_rating),
  sd_normalized_rating = sd(normalized_rating),
  favorite_threshold = 4L,
  sampled_favorites = sum(rating >= 4)
)]
fwrite(normalized_summary, file.path(output_dir, "rating_preparation_summary.csv"))
fwrite(normalization_sample[, .(rating, normalized_rating)],
       file.path(output_dir, "normalized_rating_sample.csv"))
favorite_counts <- ratings[, .(
  rated_movies = .N,
  favorites = sum(rating >= 4),
  favorite_share = mean(rating >= 4)
), by = userId]
fwrite(favorite_counts, file.path(output_dir, "favorite_counts_by_user.csv"))

# Show a small bounded ratings heatmap and user/item cosine similarity matrices.
top_item_ids <- head(movie_popularity[rating_count >= 50L, movieId], 50L)
top_user_ids <- head(ratings[movieId %in% top_item_ids, .N, by = userId][order(-N), userId], 50L)
heatmap_ratings <- ratings[userId %in% top_user_ids & movieId %in% top_item_ids]
if (nrow(heatmap_ratings) > 0L) {
  sparse_ratings <- Matrix::sparseMatrix(
    i = match(as.character(heatmap_ratings$userId), as.character(top_user_ids)),
    j = match(as.character(heatmap_ratings$movieId), as.character(top_item_ids)),
    x = heatmap_ratings$rating,
    dims = c(length(top_user_ids), length(top_item_ids)),
    dimnames = list(as.character(top_user_ids), as.character(top_item_ids))
  )
  small_matrix <- new("realRatingMatrix", data = sparse_ratings)
  png(file.path(output_dir, "ratings_heatmap.png"), width = 1000, height = 800)
  image(small_matrix, main = "Ratings heatmap: active users and popular movies")
  dev.off()

  user_similarity <- as.matrix(similarity(small_matrix, method = "cosine", which = "users"))
  movie_similarity <- as.matrix(similarity(small_matrix, method = "cosine", which = "items"))
  write.csv(user_similarity, file.path(output_dir, "user_similarity_sample.csv"))
  write.csv(movie_similarity, file.path(output_dir, "movie_similarity_sample.csv"))
  png(file.path(output_dir, "similarity_heatmaps.png"), width = 1400, height = 650)
  par(mfrow = c(1, 2), mar = c(3, 3, 4, 2))
  image(user_similarity, main = "User cosine similarity", axes = FALSE)
  image(movie_similarity, main = "Movie cosine similarity", axes = FALSE)
  dev.off()
}

cat("Analysis complete. Key outputs are in outputs/ (ignored by Git).\n")
print(rating_summary)
