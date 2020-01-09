################################
# Work with Text Mining and Sentiment Analysis
################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidytext)) install.packages("tidytext", repos = "http://cran.us.r-project.org")
if(!require(textdata)) install.packages("textdata", repos = "http://cran.us.r-project.org")
if(!require(tm)) install.packages("tm", repos = "http://cran.us.r-project.org")
if(!require(SnowballC)) install.packages("SnowballC", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(wordcloud)) install.packages("wordcloud", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

poem <- c("Roses are red,", "Violets are blue,","Sugar is sweet,", "And so are you.")
example <- tibble(line = c(1, 2, 3, 4),text = poem)
example

example_words <- example %>% unnest_tokens(word, text)

  example_words %>% 
    count(word) %>%
    arrange(desc(n))

nrc <- get_sentiments("nrc") %>%
  select(word, sentiment)
nrc

example_words %>% inner_join(nrc, by = "word") %>% select(word, sentiment)

################################
# Text Mining with the tm package
################################

filePath <- "http://www.sthda.com/sthda/RDoc/example-files/martin-luther-king-i-have-a-dream-speech.txt"
text <- readLines(filePath)
docs <- Corpus(VectorSource(text))

inspect(docs)

# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove your own stop word
# specify your stopwords as a character vector
docs <- tm_map(docs, removeWords, c("Georgia", "Mississippi")) 
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
# Text stemming
getTransformations()

writeLines(as.character(docs[[2]]))
docs <- tm_map(docs, stemDocument)
writeLines(as.character(docs[[2]]))

dcm <- DocumentTermMatrix(docs)
freq <- findFreqTerms(dcm, lowfreq = 5)
freq

tfreq <- colSums(as.matrix(dcm))
wordcloud(names(tfreq),tfreq, min.freq=5)

################################
# Wrangling sample twitter data of Apple tweets
################################
#Set up so that both R and RStudio share the same working directory.
#Change this appropriately on Mac/Linux or just remove altogether
setwd("c:\\temp")
dl <- "train.csv"

# Check for download. If exists, just load it, else download, process and save it.
if(!file.exists(dl))
{
  download.file("https://raw.githubusercontent.com/srivatsapraveen/HDXCapstone/master/02_SentimentAnalysis/data/train.csv", dl)
}

tweets <- read.csv(dl, stringsAsFactors = FALSE)
tweets %>% group_by(sentiment) %>% summarize(n())

tweets <- tweets %>% mutate(senti_text = ifelse(sentiment > 3,"P","N")) 
tweets$senti_text <- as.factor(tweets$senti_text)
tibble(tweets)
tweets %>% group_by(senti_text) %>% summarize(n())


################################
# Text Mining with the tm package
################################

tweet_docs <- VCorpus(VectorSource(tweets$text))
tweet_docs <- tm_map(tweet_docs, PlainTextDocument)
tweet_docs <- tm_map(tweet_docs, removePunctuation)
tweet_docs <- tm_map(tweet_docs, removeWords, c("apple", stopwords("english")))
tweet_docs <- tm_map(tweet_docs, content_transformer(tolower))
tweet_docs <- tm_map(tweet_docs, stemDocument)

tweet_docMatrix <- DocumentTermMatrix(tweet_docs)
wordfreq <- findFreqTerms(tweet_docMatrix, lowfreq = 25)

totfreq <- colSums(as.matrix(tweet_docMatrix))
freqword=data.frame(term=names(totfreq),occurrences=totfreq)
p <- ggplot(subset(freqword, totfreq>25), aes(term, occurrences))
p <- p + geom_bar(stat="identity")
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))
p

#setting the same seed each time ensures consistent look across clouds
set.seed(123, sample.kind="Rounding")
#limit words by specifying min frequency
wordcloud(names(totfreq),totfreq, min.freq=25)

################################
# Preparing tweets for machine learning
################################
freqDocs <- removeSparseTerms(tweet_docMatrix, 0.995)
freqTweets <- as.data.frame(as.matrix(freqDocs))
colnames(freqTweets) <- make.names(colnames(freqTweets))

freqTweets$sentiment <- tweets$sentiment
freqTweets$senti_text <- tweets$senti_text

set.seed(123, sample.kind="Rounding")
test_index <- createDataPartition(freqTweets$senti_text, times = 1, p = 0.7, list = FALSE)
train_set <- freqTweets %>% slice(-test_index)
test_set <- freqTweets %>% slice(test_index)

#**************************************************
# Regression
#**************************************************
lm_fit <- mutate(train_set, y = as.numeric(senti_text == "P")) %>% lm(y ~ ., data = .)
p_hat <- predict(lm_fit, test_set)
y_hat <- ifelse(p_hat == 1, "P", "N") %>% factor()
confusionMatrix(y_hat, test_set$senti_text)

confusionMatrix(y_hat, test_set$senti_text)$overall[["Accuracy"]]

#**************************************************
# KNN
#**************************************************
train_knn <- train(senti_text ~ ., method = "knn", data = train_set)
y_hat_knn <- predict(train_knn, test_set, type = "raw")
confusionMatrix(y_hat_knn, test_set$senti_text)

confusionMatrix(y_hat_knn, test_set$senti_text)$overall[["Accuracy"]]

#**************************************************
# Random Forest
#**************************************************
train_RF <- randomForest(senti_text ~ . , data = train_set)
predict_RF <- predict(train_RF, newdata = test_set)
confusionMatrix(predict_RF, test_set$senti_text)

confusionMatrix(predict_RF, test_set$senti_text)$overall["Accuracy"]

