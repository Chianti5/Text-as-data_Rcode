library(quanteda)
library(lsa)
library(tm)
library(NLP)
library(RColorBrewer)
library(wordcloud)

setwd("/Users/wenjia/Documents/3001TextAsData/TextProject/news-data")

## Build wordclouds ----

# Read two data files for NRA
NRAbefore <- list.files("NRA-before20151129/", full.names=TRUE)
news_nra_before <- lapply(NRAbefore, readLines)

NRAafter <- list.files("NRA-after20151129/",  full.names=TRUE)
news_nra_after <- lapply(NRAafter, readLines)

news_nra <- lapply(c(NRAbefore,NRAafter), readLines)

news_nra_before <- paste(news_nra[1:15], collapse=" ")
news_nra_after <- paste(news_nra[16:30], collapse=" ")

# Create DTM and preprocess
groups <- VCorpus(VectorSource(c("NRA after" = news_nra_after, 
                                 "NRA before" = news_nra_before)))
groups <- tm_map(groups, content_transformer(tolower))
groups <- tm_map(groups, removeNumbers)
groups <- tm_map(groups, removePunctuation)
groups <- tm_map(groups, removeWords, stopwords('english'))
groups <- tm_map(groups, stemDocument, language = "english")  
groups <- tm_map(groups, stripWhitespace)

dtm <- DocumentTermMatrix(groups)
## Label the two groups
dtm$dimnames$Docs = c("NRA after", "NRA before")
## Transpose matrix so that we can use it with comparison.cloud
tdm <- t(dtm)
## Compute TF-IDF transformation
tdm <- as.matrix(weightTfIdf(tdm))


## Display the two word clouds
comparison.cloud(tdm, max.words=100, colors=c("red", "blue"))



# Read two data files for CSGV
CSGVbefore <- list.files("CSGV-before20151129/", full.names=TRUE)
news_csgv_before <- lapply(CSGVbefore, readLines)

CSGVafter <- list.files("CSGV-after20151129/",  full.names=TRUE)
news_csgv_after <- lapply(CSGVafter, readLines)

news_csgv <- lapply(c(CSGVbefore,CSGVafter), readLines)

news_csgv_before <- paste(news_csgv[1:15], collapse=" ")
news_csgv_after <- paste(news_csgv[16:30], collapse=" ")

# Create DTM and preprocess
groups <- VCorpus(VectorSource(c("CSGV after" = news_csgv_after, 
                                 "CSGV before" = news_csgv_before)))
groups <- tm_map(groups, content_transformer(tolower))
groups <- tm_map(groups, removeWords, stopwords('english'))
groups <- tm_map(groups, stemDocument, language = "english")  
groups <- tm_map(groups, removeNumbers)
groups <- tm_map(groups, removePunctuation)
groups <- tm_map(groups, stripWhitespace)

dtm <- DocumentTermMatrix(groups)
## Label the two groups
dtm$dimnames$Docs = c("CSGV after", "CSGV before")
## Transpose matrix so that we can use it with comparison.cloud
tdm <- t(dtm)
## Compute TF-IDF transformation
tdm <- as.matrix(weightTfIdf(tdm))


## Display the two word clouds
comparison.cloud(tdm, max.words=100, colors=c("red", "blue"))





# Create DTM and preprocess
groups <- VCorpus(VectorSource(c("CSGV after" = news_csgv_after, 
                                 "CSGV before" = news_csgv_before,
                                 "NRA after" = news_nra_after, 
                                 "NRA before" = news_nra_before)))
groups <- tm_map(groups, content_transformer(tolower))
groups <- tm_map(groups, removeNumbers)
groups <- tm_map(groups, removeWords, stopwords('english'))
groups <- tm_map(groups, stemDocument, language = "english")  
groups <- tm_map(groups, removePunctuation)
groups <- tm_map(groups, stripWhitespace)

dtm <- DocumentTermMatrix(groups)
## Label the two groups
dtm$dimnames$Docs = c("CSGV_a", "CSGV_b","NRA_a", "NRA_b")
## Transpose matrix so that we can use it with comparison.cloud
tdm <- t(dtm)
## Compute TF-IDF transformation
tdm <- as.matrix(weightTfIdf(tdm))


## Display the two word clouds
comparison.cloud(tdm, max.words=100, colors=c("red", "orange","blue","green"))


## Supervised Learning ----





