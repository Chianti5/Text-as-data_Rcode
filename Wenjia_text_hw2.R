## Q2 -----------------------------------------

library(quanteda)
library(quantedaData)
library(tm)
library(SnowballC)
library(RTextTools)

# Read the data
setwd("/Users/chianti/Documents/3001TextAsData/Text_as_Data/HW2")
df.tweets <- read.csv("p4k_reviews.csv", stringsAsFactors = FALSE)

## 2a ----------------------------------------

# Calculate the median score
median(df.tweets$score) # Get 7.2

# Assign 'pos' and 'neg', i.e. labels,  according to scores
#df.tweets$label <- factor(df.tweets$score > 7.2, labels=c("negative", "positive"))

# Find those below the 10th percentile and those above the 90th percentile
quantile(df.tweets$score, .9)  # Get 8.2
quantile(df.tweets$score, .1)  # Get 5.4

# Assign labels,  according to scores
df.tweets$label[df.tweets$score > 8.2]  <- "anchor positive"
df.tweets$label[df.tweets$score > 7.2 & df.tweets$score <= 8.2]  <- "positive"
df.tweets$label[df.tweets$score >= 5.4 & df.tweets$score <= 7.2]  <- "negative"
df.tweets$label[df.tweets$score < 5.4]  <- "anchor negative"

## 2b ----------------------------------------

# Read two dictionaries for pos / neg words
neg_words <- unlist(strsplit(readLines("negative-words.txt"), " "))
pos_words <- unlist(strsplit(readLines("positive-words.txt"), " "))


# Create DTM and preprocess
gc()
groups <- VCorpus(VectorSource(df.tweets$text))
groups <- tm_map(groups, content_transformer(tolower))
groups <- tm_map(groups, removePunctuation)
groups <- tm_map(groups, stripWhitespace)

# Count the number of negative words for each row
dtm <- DocumentTermMatrix(groups, list(dictionary = neg_words))
dtm2 <- as.matrix(dtm)
df.tweets$neg_word_num <- rowSums(dtm2)

# Count the number of positive words for each row
dtm <- DocumentTermMatrix(groups, list(dictionary = pos_words))
dtm2 <- as.matrix(dtm)
df.tweets$pos_word_num <- rowSums(dtm2)

# Generate a sentiment score for each review based on the number of positive words 
# minus the number of negative words
df.tweets$pos_minus_neg <- df.tweets$pos_word_num - df.tweets$neg_word_num
df.tweets$sentiment[df.tweets$pos_minus_neg >= 0]  <- "positive"
df.tweets$sentiment[df.tweets$pos_minus_neg < 0]  <- "negative"


## 2b1 ----------------------------------------

# Find the median value of the sentiment of the reviews
median(df.tweets$pos_minus_neg) # Get 6

# Find the percentage of reviews that have a “positive” sentiment score
pos_num <- length(df.tweets$sentiment[df.tweets$sentiment == "positive"]) # Get 7175
pos_num / length(df.tweets$sentiment)

## 2b2 ----------------------------------------
# Calculate TP(true positive), TN, FP, FN 
TP <- length(df.tweets$score[df.tweets$score > 7.2 & df.tweets$sentiment == "positive"])   
TP
FN <- length(df.tweets$score[df.tweets$score > 7.2 & df.tweets$sentiment == "negative"])   
FN
FP <- length(df.tweets$score[df.tweets$score <= 7.2 & df.tweets$sentiment == "positive"])   
FP
TN <- length(df.tweets$score[df.tweets$score <= 7.2 & df.tweets$sentiment == "negative"])   
TN

# Calculate accuracy, precision, recall
accuracy <- (TP + TN) / 10000
accuracy
precision <- TP / (TP + FP)
precision
recall <- TP / (TP + FN)
recall

## 2b3 ----------------------------------------

# For the sentiment score, generate a “rank” for that review
df.tweets$sentiment_rank <- rank(df.tweets$pos_minus_neg)

# For the actual score, generate a “rank” for that review
df.tweets$true_score_rank <- rank(df.tweets$score)

# Compute the sum of all of the differences in rank for each review
sum(abs(df.tweets$sentiment_rank - df.tweets$true_score_rank))

## 2c1 ----------------------------------------

# training / test split: 2:8 as required 
training_break <- as.integer(0.2*250)

# Create DFMs for training and test data
training_dfm<-dfm(df.tweets$text[1:training_break])
test_dfm<-dfm(df.tweets$text[(training_break+1):250])

# Use the “textmodel” function in quanteda to train an unsmoothed Naive Bayes classifier
df.tweets$two_class_label <- as.numeric(df.tweets$score > 7.2)
dfm<-dfm(df.tweets$text[1:250])
training_class <- factor(c(df.tweets$two_class_label[1:training_break], rep(NA, 200)), ordered=TRUE)
NBmodel <- textmodel(x=dfm, y=training_class, model="NB", smooth=0.2)
print(NBmodel, 10)

# Predict on the test data
NBpredict <- predict(NBmodel, newdata = dfm[51:250])
test_class <- factor(c(df.tweets$two_class_label[(1+training_break):250]), ordered=TRUE)
tb <- table(NBpredict$docs$predicted[51:250], test_class)
# accuracy
print((tb[1]+tb[4])/200)
# precision
print(tb[4]/(tb[2]+tb[4]))
# recall
print(tb[4]/(tb[4]+tb[3]))

# Adjust priors to improve the quality of my predictions
for (i in 1:5)
  {
  NBmodel_docfreq <- textmodel(x=dfm, y=training_class, model="NB", smooth=(i/10), prior="docfreq")
  NBpredict_docfreq <- predict(NBmodel_docfreq, newdata = dfm[51:250])
  tb <- table(NBpredict_docfreq$docs$predicted[51:250], test_class)
  print(i)
  # accuracy
  print((tb[1]+tb[4])/200)
  # precision
  print(tb[4]/(tb[2]+tb[4]))
  # recall
  print(tb[4]/(tb[4]+tb[3]))
}

NBmodel_docfreq <- textmodel(x=dfm, y=training_class, model="NB", smooth=0.2, prior="docfreq")
NBpredict_docfreq <- predict(NBmodel_docfreq, newdata = dfm[51:250])
tb <- table(NBpredict_docfreq$docs$predicted[51:250], test_class)

## 2d1 ----------------------------------------
# Select the data in “anchor negative” and “anchor positive”
selected_pos_text <- df.tweets$text[df.tweets$label == "anchor positive"]
selected_neg_text <- df.tweets$text[df.tweets$label == "anchor negative"]
selected_dfm <- dfm(c(selected_pos_text, selected_neg_text))

# Calculate the relative frequency
rowsum <- rowSums(selected_dfm) 

# Note that there are three documents that have 0 rowSums(selected_dfm) 
which(rowSums(selected_dfm) == 0 ) # Get 822, 1610 1760
rowsum[822] <- 1
rowsum[1610] <- 1
rowsum[1760] <- 1
relative_freq_matrix <- selected_dfm/ rowsum # This is the F matrix in the paper

# Find the words that occure less than 10 times in the data
ignored_words <- names(which(colSums(selected_dfm) < 10))

# Calculate the probility that an occurrence of word w implies that we are reading text r
# This is the P matrix in the paper
word_prob_matrix <- sweep(relative_freq_matrix,2,colSums(relative_freq_matrix),`/`)


# Calculate the score
pos_length <- length(selected_pos_text)
all_length <- pos_length + length(selected_neg_text)

# Only consider the words that occure >= 10 times in the data
score <- score[!names(score) %in% ignored_words]

# Continue calculating the score
score <- colSums(word_prob_matrix[1:pos_length,]) - colSums(word_prob_matrix[(pos_length+1):all_length,])
lowest_scores <- quantile(score, .05)
highest_scores <- quantile(score, .95)
score[score > highest_scores]
score[score < lowest_scores]

## 2d2 ----------------------------------------
ls <- as.list(score[score == 1 ])
wordscore_pos_dic <- unlist(names(ls))
ls <- as.list(score[score == -1 ])
wordscore_neg_dic <- unlist(names(ls))


# Create DTM and preprocess
gc()
groups <- VCorpus(VectorSource(df.tweets$text))
groups <- tm_map(groups,
                 content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')),
                 mc.cores=1)
groups <- tm_map(groups, content_transformer(tolower))
groups <- tm_map(groups, removePunctuation)
groups <- tm_map(groups, stripWhitespace)

# Only consider the words appeared in wordscore_dic
dtm_wordscore_pos <- DocumentTermMatrix(groups, list(dictionary = wordscore_pos_dic))
dtm_wordscore_pos <- as.matrix(dtm_wordscore_pos)
dtm_wordscor_neg <- DocumentTermMatrix(groups, list(dictionary = wordscore_neg_dic))
dtm_wordscore_neg <- as.matrix(dtm_wordscore_neg)

# Somehow I kept getting an error running DocumentTermMatrix, so I used an alternative
# method to calculate the dfm
dtm_wordscore_pos <- dfm(df.tweets$text, verbose = TRUE, toLower = TRUE, removePunct = TRUE, 
                         removeSeparators = FALSE, keptFeatures = wordscore_pos_dic)
dtm_wordscore_pos <- as.matrix(dtm_wordscore_pos)
dtm_wordscore_neg <- dfm(df.tweets$text, verbose = TRUE, toLower = TRUE, removePunct = TRUE, 
                        removeSeparators = FALSE, keptFeatures = wordscore_neg_dic)
dtm_wordscore_neg <- as.matrix(dtm_wordscore_neg)

df.tweets$total_pos_wordscore <- rowSums(dtm_wordscore_pos)
df.tweets$total_neg_wordscore <- rowSums(dtm_wordscore_neg)

df.tweets$wordscore_pos_minus_neg <- df.tweets$total_pos_wordscore-df.tweets$total_neg_wordscore

# For the sentiment score, generate a “rank” for that review
df.tweets$sentiment_rank <- rank(df.tweets$wordscore_pos_minus_neg)

# For the actual score, generate a “rank” for that review
df.tweets$true_score_rank <- rank(df.tweets$score)

# Compute the sum of all of the differences in rank for each review
sum(abs(df.tweets$sentiment_rank - df.tweets$true_score_rank))

## 2e ----------------------------------------
# Train an SVM
dtm  <- create_matrix(df.tweets$text[1:1000], language="english", stemWords = FALSE,
                      weighting = weightTfIdf, removePunctuation = FALSE)


linearkernel_accuracy = list()

for (i in 1:9) {
  training_break = i * 100
  container <- create_container(dtm, t(df.tweets$two_class_label ), trainSize=1:training_break,
                                testSize=training_break:1000, virgin=FALSE)
  cv.svm <- cross_validate(container, nfold=5, algorithm = 'SVM', kernel = 'linear')
  print(i)
  print(cv.svm$meanAccuracy)
  linearkernel_accuracy <- c(linearkernel_accuracy, cv.svm$meanAccuracy)
}
gc()

radialkernel_accuracy <- list()
for (i in 1:9) {
  training_break = i * 100
  container <- create_container(dtm, t(df.tweets$two_class_label ), trainSize=1:training_break,
                                testSize=training_break:1000, virgin=FALSE)
  cv.svm <- cross_validate(container, nfold=5, algorithm = 'SVM', kernel = "radial")
  print(i)
  print(cv.svm$meanAccuracy)
  radialkernel_accuracy <- c(radialkernel_accuracy, cv.svm$meanAccuracy)
}

# Plot the accuracy using two different kernels
training_size <- c(.1, .2, .3, .4, .5, .6, .7, .8, .9)

# Graph linearkernel_accuracy  
plot(training_size, linearkernel_accuracy, type="o", col="blue" , ylim=c(.5, .67),
     ann=FALSE)

# Graph radialkernel_accuracy with red dashed line and square points
lines(training_size, radialkernel_accuracy, type="o", col="red")

# Create a title, labels and legends
title(main="Comparing two kernels", col.main="red", font.main=4)
title(ylab="Accuracy", col.lab=rgb(0,0.5,0))
title(xlab="Training size", col.lab=rgb(0,0.5,0))
legend(1,  c("linear kernel","radial kernel"), cex=0.8, 
       col=c("blue","red"), pch=21:22, lty=1:2)

## 3a ----------------------------------------
# Read the data
df.trustworthiness <- read.csv("CF_rate_trustworthiness.csv", stringsAsFactors = FALSE)

# Note that we are only interested in the following three columns
# df.trustworthiness$rating
# df.trustworthiness$image_name
# df.trustworthiness$X_country

# Set the target - blackman / whiteman / whitewomen 
df.trustworthiness$target <- gsub('[[:digit:]]+', '', df.trustworthiness$image_name)

# First I found that line 34 was empty for X_country after running the following code
which(unique(df.trustworthiness$X_country) == "")
which(unique(df.trustworthiness$rating) == "")
which(unique(df.trustworthiness$image_name) == "")

country_data <- list( )
all_val <- c()
country_group <- factor()
for (i in unique(unlist(df.trustworthiness$X_country))){
  if (i != ""){
    if (length(df.trustworthiness$rating[df.trustworthiness$X_country == i]) > 1) {
      country_data <- append(country_data, list(df.trustworthiness$rating[df.trustworthiness$X_country == i]))
      country_group <- append(country_group, i)
      all_val <- c(all_val, df.trustworthiness$rating[df.trustworthiness$X_country == i])
    }
    }
}

# Create a vector storing the ratings
dat <- c()
for (j in 1:30){
  dat <- c(dat, c(country_data[j]))
}

# Run ANOVA
bartlett.test(dat, country_group)

# Run t-test
higher_rating_country <- list()
for (j in 1:30){
  #print(country_group[j])
  t_test <- t.test(unlist(dat[j]), all_val, alternative="greater")
  #print(t_test$p.value)
  if (t_test$p.value < .05) {
    print(c(country_group[j], "people give significantly higher ratings"))
  }
}

# 3b --------------------------------------

demographic_data <- list( )
demographic_group <- factor()
for (i in unique(unlist(df.trustworthiness$target))){
  if (i != ""){
    if (length(df.trustworthiness$rating[df.trustworthiness$target == i]) > 1) {
      demographic_data <- append(demographic_data, list(df.trustworthiness$rating[df.trustworthiness$target == i]))
      demographic_group <- append(demographic_group, i)
    }
  }
}

all_demographic_data <- c(unlist(demographic_data[1]),unlist(demographic_data[2]),
                          unlist(demographic_data[3]))


# Run t-test
higher_rating_demographic <- list()
for (j in 1:3){
  print(demographic_group[j])
  t_test <- t.test(unlist(demographic_data[j]), all_demographic_data)
  print(t_test$p.value)
  if (t_test$p.value < .05) {
    print(c(demographic_group[j], "people give significantly higher ratings"))
  }
}
