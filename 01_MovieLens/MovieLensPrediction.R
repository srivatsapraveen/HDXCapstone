################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

options(digits = 8)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

#Set the working directory. This ensures that both R and RStudio use the same working folder for the files.NOTE : This works on Windows, change it for Mac/Linux.
setwd("c:\\temp")
edxfile <- "edx.file"

# Check for download. If exists, just load it, else download, process and save it.
if(!file.exists(edxfile))
{
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

  movielens <- left_join(ratings, movies, by = "movieId")

  # Validation set will be 10% of MovieLens data

  set.seed(1, sample.kind="Rounding")
  # if using R 3.5 or earlier, use `set.seed(1)` instead
  
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]

  # Make sure userId and movieId in validation set are also in edx set
  validation <- temp %>% 
     semi_join(edx, by = "movieId") %>%
     semi_join(edx, by = "userId")

  # Add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)

  rm(dl, ratings, movies, test_index, temp, movielens, removed)
  save(edx, validation, file = edxfile)
} else {
  load(edxfile)
}

################################
# Explore the data
################################

# see 'edx' data in tidy format
edx %>% as_tibble()

# number of unique users and movies
edx %>% summarize(total_users = n_distinct(userId), 
                  total_movies = n_distinct(movieId),
                  total_ratings = n())

count_byrating <- edx %>% group_by(rating) %>% summarize(count = n())
count_byrating %>% ggplot(aes(x=rating,y=count)) +
  geom_bar( aes(y = count), stat = "identity", position = "identity") +
  xlab("Rating") +
  ylab("Count") +
  ggtitle("Ratings Histogram") +
  theme(plot.title = element_text(hjust = 0.5))

edx %>% 
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black") +
  scale_x_log10() +
  xlab("# Ratings") +
  ylab("# Users") +
  ggtitle("Users Number of Ratings") +
  theme(plot.title = element_text(hjust = 0.5))

################################
# Compute the rmse function
################################

# rmsefunction for vectors of ratings and their corresponding predictors
getrmse <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

################################
# Approach 1: predict same rating for all movies regardless of movies or users
################################
edx_mean <- mean(edx$rating)
edx_mean

rmse01 <- getrmse(validation$rating, edx_mean)

# add rmse results in a table
all_rmses <- data_frame(method = "(rmse01) Simple approach", rmse = rmse01)
all_rmses %>% knitr::kable()

################################
# Approach 2: Take into effect the movie effect.
################################
avg_bymovie <- edx %>%
  group_by(movieId) %>%
  summarise(mov_mean = mean(rating - edx_mean))

mean_bymovie <- edx_mean + validation %>%
  left_join(avg_bymovie, by='movieId') %>%
  pull(mov_mean)

rmse02 <- getrmse(validation$rating,mean_bymovie)

all_rmses <- bind_rows(all_rmses,data_frame(method="(rmse02) Movie Effect", rmse = rmse02))
all_rmses %>% knitr::kable()

################################
# Approach 3: Take into effect the user effect.
################################
avg_byuser <- edx %>%
  left_join(avg_bymovie, by="movieId") %>%
  group_by(userId) %>%
  summarise(user_mean = mean(rating - edx_mean - mov_mean))

mean_byuser <- validation %>%
  left_join(avg_bymovie, by='movieId') %>%
  left_join(avg_byuser, by='userId') %>%
  mutate(user_mean = edx_mean + mov_mean + user_mean) %>%
  pull(user_mean)

rmse03 <- getrmse(validation$rating,mean_byuser)

all_rmses <- bind_rows(all_rmses,data_frame(method="(rmse03) User Effect", rmse = rmse03))
all_rmses %>% knitr::kable()


################################
# Approach 4: Regularizing the effects
################################
lambdas <- seq(0, 10, 0.5)

reg_means <- sapply(lambdas, function(l){
  
  ravg_mov <- edx %>% 
    group_by(movieId) %>%
    summarize(rmov_mean = sum(rating - edx_mean)/(n()+l))
  
  ravg_user <- edx %>% 
    left_join(ravg_mov, by="movieId") %>%
    group_by(userId) %>%
    summarize(ruser_mean = sum(rating - rmov_mean - edx_mean)/(n()+l))
  
  rmean_regularized <- 
      validation %>% 
      left_join(ravg_mov, by = "movieId") %>%
      left_join(ravg_user, by = "userId") %>%
      mutate(reg_mean = edx_mean + rmov_mean + ruser_mean) %>%
      pull(reg_mean)
  
  return(getrmse(validation$rating,rmean_regularized))
})

qplot(lambdas, reg_means) 

# lambda that minimizes rmse
lambda <- lambdas[which.min(reg_means)]

# calculate rmse after regularizing movie + user effect from previous models
rmse04 <- min(reg_means)

all_rmses <- bind_rows(all_rmses,data_frame(method="(rmse04) Regularized Effect", rmse = rmse04))
all_rmses %>% knitr::kable()

################################
# Approach 5: Take into effect the genre effect.
################################
avg_bygenres <- edx %>%
  left_join(avg_bymovie, by="movieId") %>%
  left_join(avg_byuser, by="userId") %>%
  group_by(genres) %>%
  summarise(genres_mean = mean(rating - edx_mean - mov_mean - user_mean))

mean_bygenre <- validation %>%
  left_join(avg_bymovie, by='movieId') %>%
  left_join(avg_byuser, by='userId') %>%
  left_join(avg_bygenres, by='genres') %>%
  mutate(genres_mean = edx_mean + mov_mean + user_mean + genres_mean) %>%
  pull(genres_mean)

rmse05 <- getrmse(validation$rating,mean_bygenre)

all_rmses <- bind_rows(all_rmses,data_frame(method="(rmse05) Genre Effect", rmse = rmse05))
all_rmses %>% knitr::kable()

################################
# Approach 6: Take into effect the genre effect with independent genres
################################
setwd("c:\\temp")
edxfile_g <- "edx_g.file"
if(!file.exists(edxfile)){
	validation_g <- validation %>%  separate_rows(genres, sep = "\\|")
	edx_g <- edx %>% separate_rows(genres, sep = "\\|")
	save(edx_g, validation_g, file = edxfile_g)
} else {
  	load(edxfile_g)
}

# number of unique users and movies
edx_g %>% summarize(total_users = n_distinct(userId), 
                    total_movies = n_distinct(movieId), 
                    total_genres = n_distinct(genres))
edx_g %>% group_by(genres) %>% summarize(count = n())

avg_byg <- edx_g %>%
  left_join(avg_bymovie, by="movieId") %>%
  left_join(avg_byuser, by="userId") %>%
  group_by(genres) %>%
  summarise(genres_mean = mean(rating - edx_mean - mov_mean - user_mean))

mean_byg <- validation_g %>%
  left_join(avg_bymovie, by='movieId') %>%
  left_join(avg_byuser, by='userId') %>%
  left_join(avg_bygenres, by='genres') %>%
  mutate(genres_mean = edx_mean + mov_mean + user_mean + genres_mean) %>%
  pull(genres_mean)

rmse06 <- getrmse(validation_g$rating,mean_byg)

all_rmses <- bind_rows(all_rmses,data_frame(method="(rmse06) Split Genre Effect", rmse = rmse06))
all_rmses %>% knitr::kable()

################################
# Approach 7: Regularizing the effects with independent genres
################################
lambdas <- seq(0, 20, 1)

reg_meang <- sapply(lambdas, function(l){
  
  ravg_movg <- edx_g %>% 
    group_by(movieId) %>%
    summarize(rmov_meang = sum(rating - edx_mean)/(n()+l))
  
  ravg_userg <- edx_g %>% 
    left_join(ravg_movg, by="movieId") %>%
    group_by(userId) %>%
    summarize(ruser_meang = sum(rating - rmov_meang - edx_mean)/(n()+l))
  
  ravg_byg <- edx_g %>%
    left_join(ravg_movg, by="movieId") %>%
    left_join(ravg_userg, by="userId") %>%
    group_by(genres) %>%
    summarise(rgenres_meang = mean(rating - rmov_meang - ruser_meang - edx_mean))

  rmean_regularizedg <- validation_g %>%
    left_join(ravg_movg, by='movieId') %>%
    left_join(ravg_userg, by='userId') %>%
    left_join(ravg_byg, by='genres') %>%
    mutate(reg_meang = edx_mean + rmov_meang + ruser_meang + rgenres_meang) %>%
    pull(reg_meang)

  return(getrmse(validation_g$rating,rmean_regularizedg))
})

qplot(lambdas, reg_meang) 

# lambda that minimizes rmse
lambda <- lambdas[which.min(reg_meang)]

# calculate rmse after regularizing movie + user effect from previous models
rmse07 <- min(reg_meang)

all_rmses <- bind_rows(all_rmses,data_frame(method="(rmse07) Regularized Split Genre Effect", rmse = rmse07))
all_rmses %>% knitr::kable()

