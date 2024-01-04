library(data.table)
library(tm)
library(dplyr)
library(SnowballC)
library(ggplot2)
library(wordcloud)

train_data_path <- "/Users/pranjalpatil/Courses/CSP571/Project/drug_dataset/drugLibTrain_raw.tsv"
test_data_path <- "/Users/pranjalpatil/Courses/CSP571/Project/drug_dataset/drugLibTest_raw.tsv"

train_data <- fread(train_data_path, drop = 1)
test_data <- fread(test_data_path, drop = 1)

# Combine reviews into a single column
combine_reviews <- function(data) {
  data$reviews <- paste(data$benefitsReview, data$sideEffectsReview, data$commentsReview, sep = " ")
  return(data)
}

# Apply the function to both train and test sets
train_data <- combine_reviews(train_data)
test_data <- combine_reviews(test_data)

remove_special_characters <- function(corpus) {
  corpus_cleaned <- tm_map(corpus, content_transformer(function(x) gsub("[^a-zA-Z0-9 ]", "", x)))
  return(corpus_cleaned)
}

# Data preprocessing on train dataset
train_data_corpus = VCorpus(VectorSource(train_data$reviews))
train_data_corpus

train_data_corpus <- tm_map(train_data_corpus, content_transformer(function(x) iconv(enc2utf8(x), sub = "byte")))

train_data_corpus <- remove_special_characters(train_data_corpus)

train_data_corpus_new = tm_map(train_data_corpus, removeWords,stopwords())
as.character(train_data_corpus_new[[1]])

train_data_corpus_new = tm_map(train_data_corpus_new, stripWhitespace)
as.character(train_data_corpus_new[[1]])

train_data_corpus_new = tm_map(train_data_corpus_new, removePunctuation)
as.character(train_data_corpus_new[[1]])

train_data_corpus_new = tm_map(train_data_corpus_new, content_transformer(tolower))
as.character(train_data_corpus_new[[1]])

# Data preprocessing on test dataset
test_data_corpus = VCorpus(VectorSource(test_data$reviews))
test_data_corpus

test_data_corpus <- tm_map(test_data_corpus, content_transformer(function(x) iconv(enc2utf8(x), sub = "byte")))

test_data_corpus <- remove_special_characters(test_data_corpus)

test_data_corpus_new = tm_map(test_data_corpus, removeWords,stopwords())
as.character(test_data_corpus_new[[1]])

test_data_corpus_new = tm_map(test_data_corpus_new, stripWhitespace)
as.character(test_data_corpus_new[[1]])

test_data_corpus_new = tm_map(test_data_corpus_new, removePunctuation)
as.character(test_data_corpus_new[[1]])

test_data_corpus_new = tm_map(test_data_corpus_new, content_transformer(tolower))
as.character(test_data_corpus_new[[1]])

# Create Document-Term Matrix (DTM) with TF-IDF
dtm_tfidf_train <- DocumentTermMatrix(train_data_corpus_new, control = list(weighting = weightTfIdf))
dtm_tfidf_test <- DocumentTermMatrix(test_data_corpus_new, control = list(weighting = weightTfIdf))

# Convert DTM to a data frame for better readability
dtm_tfidf_df_train <- as.data.frame(as.matrix(dtm_tfidf_train))
dtm_tfidf_df_test <- as.data.frame(as.matrix(dtm_tfidf_test))

# Findings/insights
cat("Findings/Insights:\n")
cat("Dimensions of TF-IDF matrix for the training set:", dim(dtm_tfidf_df_train), "\n")
cat("Dimensions of TF-IDF matrix for the test set:", dim(dtm_tfidf_df_test), "\n")

# Find top terms based on TF-IDF scores
top_terms_tfidf_train <- names(sort(colSums(dtm_tfidf_df_train), decreasing = TRUE))[1:50]
top_terms_tfidf_test <- names(sort(colSums(dtm_tfidf_df_test), decreasing = TRUE))[1:50]

cat("\nTop Terms Based on TF-IDF (Training Set):\n")
print(top_terms_tfidf_train)

cat("\nTop Terms Based on TF-IDF (Test Set):\n")
print(top_terms_tfidf_test)

# Bag-of-Words
# Create Document-Term Matrix (Bag-of-Words)
dtm_train <- DocumentTermMatrix(train_data_corpus_new)
dtm_test <- DocumentTermMatrix(test_data_corpus_new)

# Findings/Insights
# Display the top terms based on document frequency
top_terms_train <- findFreqTerms(dtm_train, lowfreq = 10)[1:50]
top_terms_test <- findFreqTerms(dtm_test, lowfreq = 10)[1:50]

cat("Bag-of-Words: Top Terms in Training Set:\n")
print(top_terms_train)

cat("\nBag-of-Words: Top Terms in Test Set:\n")
print(top_terms_test)

# Word Clouds for Bag-of-Words
# Convert DTM to a data frame
dtm_train_df <- as.data.frame(as.matrix(dtm_train))
dtm_test_df <- as.data.frame(as.matrix(dtm_test))

# Add column names to the DTM data frames
colnames(dtm_train_df) <- make.names(colnames(dtm_train_df))
colnames(dtm_test_df) <- make.names(colnames(dtm_test_df))

word_freq_train <- colSums(dtm_train_df)
word_freq_test <- colSums(dtm_test_df)

# Generate Word Clouds
wordcloud(names(word_freq_train), freq = word_freq_train, scale = c(3, 0.5), max.words = 50, random.order = FALSE, colors = brewer.pal(8, "Dark2"))
title(main = "Word Cloud for Bag-of-Words in Training Set")

wordcloud(names(word_freq_test), freq = word_freq_test, scale = c(3, 0.5), max.words = 50, random.order = FALSE, colors = brewer.pal(8, "Dark2"))
title(main = "Word Cloud for Bag-of-Words in Test Set")
