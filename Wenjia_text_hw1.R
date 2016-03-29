library(quanteda)
library(quantedaData)
library(dplyr)
library(koRpus)
library(stringr)
library(lsa)
library(ggbiplot)
#install.packages("gridBase", repos="http://cran.r-project.org") 
library(gridBase)

## Question 1 ----------------------------------------------------------------------------

data("SOTUCorpus")
#sotus <- texts(SOTUCorpus)

# Extract the years of all the documents and convert them into numbers
year<-(SOTUCorpus$documents$Date)
year_num <- as.numeric(format(year, "%Y"))
# Extract the president of all the documents
pres<-(SOTUCorpus$documents$President)
# Select the subset of SOTUCorpus as the data for question 1
data <- subset(SOTUCorpus, year_num >= 2009 & year_num <= 2016, pres == "Obama" )

## Calculate TTR of each of the speech
tokens<-tokenize(data, removePunct=TRUE)

tokenz<-lapply(tokens,  length )

typez<-lapply(lapply(tokens,  unique ), length)

TTRz<-mapply("/",typez,tokenz,SIMPLIFY = FALSE)
print(unlist(TTRz))

## Create a document feature matrix of the two speeches, with no pre-processing other than 
## to remove the punctuation
two_data <- subset(SOTUCorpus, year_num == 2016|year_num == 2015, pres == "Obama")
two_dfm <- dfm(two_data, verbose = FALSE, toLower = FALSE,
               removeNumbers = FALSE, removeSeparators = FALSE)

## Calculate the cosine distance between the two documents with quanteda
cos_similarity <- similarity(two_dfm, margin = "documents")
as.matrix(cos_similarity)

## Consider stemming the words and repeat the above procedures
stems<-wordstem(tokens)
tokenz<-lapply(stems,  length )
typez<-lapply(lapply(stems,  unique ), length)
TTRz_stemming<-mapply("/",typez,tokenz,SIMPLIFY = FALSE)
print(unlist(TTRz_stemming))

stemmed_two_dfm <- dfm(two_data, removePunct = TRUE, stem=TRUE, 
                       verbose = FALSE, toLower = FALSE,
                       removeNumbers = FALSE, removeSeparators = FALSE)
cos_similarity_stemmed <- similarity(stemmed_two_dfm, margin = "documents")
as.matrix(cos_similarity_stemmed)

## Consider removing stop words and repeat the above procedures
data_without_stopwords <- removeFeatures(tokenize(data, removePunct = TRUE), stopwords("english"))
tokenz<-lapply(data_without_stopwords,  length )
typez<-lapply(lapply(data_without_stopwords,  unique ), length)
TTRz_no_stopwords<-mapply("/",typez,tokenz,SIMPLIFY = FALSE)
print(unlist(TTRz_no_stopwords))

two_dfm_no_stopwords <- dfm(two_data, removePunct = TRUE, 
                            ignoredFeatures = stopwords("english"),
                            verbose = FALSE, toLower = FALSE,
                            removeNumbers = FALSE, removeSeparators = FALSE)
cos_similarity_without_stopwords <- similarity(two_dfm_no_stopwords, margin = "documents")
as.matrix(cos_similarity_without_stopwords)

## Consider converting all words to lowercase and repeat the above procedures
lowercase_tokens<-toLower(tokens)
tokenz<-lapply(lowercase_tokens,  length )
typez<-lapply(lapply(lowercase_tokens,  unique ), length)
TTRz_lowercase<-mapply("/",typez,tokenz,SIMPLIFY = FALSE)
print(unlist(TTRz_lowercase))

lowercase_two_dfm <- dfm(two_data, removePunct = TRUE, toLower = TRUE,
                         verbose = FALSE, 
                         removeNumbers = FALSE, removeSeparators = FALSE)
cos_similarity_lowercase <- similarity(lowercase_two_dfm, margin = "documents")
as.matrix(cos_similarity_lowercase)

## Consider tf-idf
lower_tokens_no_stopwords <- toLower(data_without_stopwords)
lower_no_stopwords_dfm <- dfm(lower_tokens_no_stopwords)
normalized<-tfidf(lower_no_stopwords_dfm, normalize=TRUE)
topfeatures(normalized, 20)

## Calculate the MTLD of each speech, with the TTR limit set at .72

text_data <- texts(data)
MTLD_data <- tokenize(text_data, format='obj', lang = 'en')
gc()
lex.div(MTLD_data, measure = "MTLD", char = "MTLD")

## Question 2 ----------------------------------------------------------------------------
sentence_a <- "The tip of the tongue taking a trip of three steps down the palate to tap, at three, on the teeth."
sentence_b <- "Kevin tripped and chipped his tooth on the way to platform number three."

# Tokenize two sentences by hand after setting all the words to lower case
a_tokens <- unlist(strsplit(tolower(sentence_a)," "))
b_tokens <- unlist(strsplit(tolower(sentence_b)," "))
# Remove punctuations in two tokenized lists
a_tokens <- gsub("[[:punct:]]", "", a_tokens)
b_tokens <- gsub("[[:punct:]]", "", b_tokens)
wordlist <- unique(unlist(c(a_tokens, b_tokens)))
print(wordlist)

# Calculate DFMs for two sentences
dfm_a <- sapply(wordlist, function(x) sum(str_count(x, a_tokens)))
dfm_b <- sapply(wordlist, function(x) sum(str_count(x, b_tokens)))

# Calculate the Euclidean distance between the two sentences
sqrt(sum((dfm_a - dfm_b)^2))

# Calculate the Manhattan distance between the two sentences
sum(abs(dfm_a - dfm_b))

# Calculate the cosine similarity of the two sentences
norm_vec <- function(x) sqrt(sum(x^2))
dfm_a %*% dfm_b / (norm_vec(dfm_a)*norm_vec(dfm_b))

## Question 3 ----------------------------------------------------------------------------

##3b

# Read the function words 
setwd("/Users/chianti/Documents/3001TextAsData/")
function_word <- unlist(strsplit(readLines("function_words.txt"), " "))
function_word  


## 3c

setwd("/Users/chianti/Documents/3001TextAsData/3001Text-HW1/dickens_austen/")
files <- list.files( full.names=TRUE)
text <- lapply(files, readLines)
text<-unlist(lapply(text, function(x) paste(x, collapse = " ")))
files
gc()

# Initiate a list to store the divided data
divided_data <- list()

# First concatenate the texts for each auther
austen <- paste(text[1], text[2], text[3], text[4], text[5], " ")
dickens <- paste(text[6], text[7], text[8], text[9], text[10]," ")
mystery <- text[11]

# Initiate two lists to store the divided text
austen_block <- character()
dickens_block <- character()
mystery_block <- character()
gc()

# Divide austen's words
total_num_austen <- sapply(gregexpr("\\W+", austen), length) + 1
print(total_num_austen)
# The next for loop divide each author’s work into 1,700 words
block_num_austen <- total_num_austen/1700
contents <- unlist(strsplit(austen, "[ ]"))
for (j in 1:floor(block_num_austen)){
  austen_block[j] <- paste(contents[((j-1)*1700): (j*1700)], collapse = " ")
}
austen_block[j] <-paste(contents[j*1700:total_num_austen], collapse = " ")
gc()

# Divide dickens' words
total_num_dickens <- sapply(gregexpr("\\W+", dickens), length) + 1
print(total_num_dickens)
# The next for loop divide each author’s work into 1,700 words
block_num_dickens <- total_num_dickens/1700
contents_d <- unlist(strsplit(dickens, "[ ]"))
for (j in 1:floor(block_num_dickens)){
  dickens_block[j] <- paste(contents_d[((j-1)*1700): (j*1700)], collapse = " ")
}
dickens_block[j] <-paste(contents_d[j*1700:total_num_dickens], collapse = " ")
gc()

austen_dfm <- dfm(austen_block, keptFeatures =function_word)
dickens_dfm <- dfm(dickens_block, keptFeatures =function_word)


# Divide Mr. mystery' words
total_num_mystery <- sapply(gregexpr("\\W+", mystery), length) + 1
print(total_num_mystery)
# The next for loop divide each author’s work into 1,700 words
block_num_mystery <- total_num_mystery/1700
contents_m <- unlist(strsplit(mystery, "[ ]"))
for (j in 1:floor(block_num_mystery)){
  mystery_block[j] <- paste(contents_m[((j-1)*1700): (j*1700)], collapse = " ")
}
mystery_block[j] <-paste(contents_m[(j*1700):total_num_mystery], collapse = " ")
mystery_dfm <- dfm(mystery_block, keptFeatures =function_word)


## 3d & 3e

snippets_pca_austen <-prcomp(austen_dfm, center=TRUE, scale.=TRUE)
snippets_pca_diskens <-prcomp(dickens_dfm, center=TRUE, scale.=TRUE)

all_block <- c(austen_block, dickens_block)
all_dfm <- dfm(all_block, keptFeatures =function_word)
snippets_pca <-prcomp(all_dfm, center=TRUE, scale.=TRUE)


##Predict : input the dfm (with appropriate features) of the mystery text
predicted<-predict(snippets_pca, newdata=mystery_dfm)
predicted

##Fisher's linear discrimination rule: choose the group that has a closer group mean; just 2 dimensions
d<-length(dickens_block)
a<-length(austen_block)

#find the mean of the first two PCs 
austen_pc1_mean<-mean(snippets_pca$x[1:a,1])
austen_pc2_mean<-mean(snippets_pca$x[1:a,2])

austen_mean <- c(austen_pc1_mean, austen_pc2_mean)
austen_mean

dickens_pc1_mean<-mean(snippets_pca$x[a:(a+d),1])
dickens_pc2_mean<-mean(snippets_pca$x[a:(a+d),2])
dickens_mean<-c(dickens_pc1_mean, dickens_pc2_mean)
dickens_mean

mystery_pc1_mean<-mean(predicted[,1])
mystery_pc2_mean<-mean(predicted[,2])
mystery_mean<-c(mystery_pc1_mean, mystery_pc2_mean)
mystery_mean

dist(rbind(mystery_mean, austen_mean))
dist(rbind(mystery_mean, dickens_mean))

# Find the top features for Austen
loadings_abs <- abs(snippets_pca_austen$rotation)
df <- data.frame(loadings_abs)
PC1_df <- df["PC1"]
selected_df <- subset(PC1_df, PC1 > 0.18)
selected_df

# Find the top features for Dickens
loadings_abs <- abs(snippets_pca_diskens$rotation)
df <- data.frame(loadings_abs)
PC1_df <- df["PC1"]
selected_df <- subset(PC1_df, PC1 > 0.16)
selected_df


# Plot
plot(snippets_pca, type = "l")
authors <- c(rep("Austen", a), rep("Dickens", d))

g <- ggbiplot(snippets_pca, obs.scale = 1, var.scale = 1, 
              groups = authors)
g<- g + theme(legend.direction = 'horizontal', 
              legend.position = 'top')
g


## Question 4 ----------------------------------------------------------------------------
# Zipf's law
gc()
all_text <- paste0(paste0(austen,dickens)[1],paste0(austen,dickens)[1])
one_dfm <- dfm(all_text)

#n = 31104
n = 100
sorted_dfm <- topfeatures(one_dfm,n)
freq <- as.numeric(sorted_dfm)  

plot(log10(1:n), log10(topfeatures(one_dfm, n)),
     xlab="log10(rank)", ylab="log10(frequency)", main="Top 1000 Words")
# regression to check if slope is approx -1.0
regression <- lm(log10(sorted_dfm) ~ log10(1:n))
abline(regression, col="red")
confint(regression)

# Heap's law

tokens<-tokenize(all_text, removePunct=TRUE) 
Tee<-lapply(tokens,  length )
Tee<-sum(unlist(Tee))

mydfm <- one_dfm
M<-length(mydfm@Dimnames$features)

k<- 44
b<-.435

k * (Tee)^b
M


## Question 5 ----------------------------------------------------------------------------
Pride_text <- text[4]
Tale_text <- text[10]


kwic(Pride_text, "society", 5)
kwic(Tale_text, "society", 5)

kwic(Pride_text, "social", 5)
kwic(Tale_text, "social", 5)

kwic(Pride_text, "world", 3)
kwic(Tale_text, "world", 3)

kwic(Pride_text, "British", 4)
kwic(Tale_text, "British", 4)

## Question 6 ----------------------------------------------------------------------------
setwd("/Users/chianti/Documents/3001TextAsData/3001Text-HW1/cons/")
cons_files <- list.files( full.names=TRUE)
cons_text <- lapply(cons_files, readLines)
cons_text<-unlist(lapply(cons_text, function(x) paste(x, collapse = " ")))
cons_files<-gsub("./Con", "", cons_files )
cons_files<-gsub(".txt", "", cons_files )
cons_files
gc()

# Load packages
#install.packages("openNLP")
require(openNLP)

#define the standard error function
std <- function(x) sd(x)/sqrt(length(x))

n <- length(cons_files)

# Initialize data frames
year_FRE<-data.frame(matrix(ncol = n, nrow = 100))

# Use Maxent_Sent_Token_Annotator to break each txt file into sentences
sent_token_annotator <- Maxent_Sent_Token_Annotator()
sent_token_annotator

# This for loop split each txt file into sentences
for (i in 1:n){
  
  a1 <- annotate(cons_text[i], sent_token_annotator)
  input_data <- as.String(cons_text[i])
  input_data[a1]
  
  df<-data.frame(text=input_data[a1],year=cons_files[i])
  gc()

}


# run the bootstraps
for(j in 1:100){
  
  #sample 3000
  bootstrapped<-sample_n(df, 3000, replace=TRUE)
  bootstrapped$read_FRE<-readability(as.character(bootstrapped$text), "Flesch")
  
  #store results
  year_FRE[i,]<-aggregate(bootstrapped$read_FRE, by=list(bootstrapped$year), FUN=mean)[,2]
  
}

year_ses<-apply(year_FRE, 2, std)
year_means<-apply(year_FRE, 2, mean)





###Plot results--year

coefs<-year_means
ses<-year_ses

y.axis <- c(1:23)*0.27
min <- min(coefs - 2*ses)
max <- max(coefs + 2*ses)
var.names <- colnames(year_FRE)
adjust <- 0
par(mar=c(2,8,2,2))

plot(coefs, y.axis, type = "p", axes = F, xlab = "", ylab = "", pch = 19, cex = .8,
     xlim=c(min,max),ylim = c(.5,6.5), main = "")
for (i in 1:23){
  rect(min,.4*(i-1),max,.4*i, col = c("grey97"), border="grey90", lty = 2)
}

axis(1, at = seq(min,max,(max-min)/10),
     labels = c(round(min+0*((max-min)/10),3),
                round(min+1*((max-min)/10),3),
                round(min+2*((max-min)/10),3),
                round(min+3*((max-min)/10),3),
                round(min+4*((max-min)/10),3),
                round(min+5*((max-min)/10),3),
                round(min+6*((max-min)/10),3),
                round(min+7*((max-min)/10),3),
                round(min+8*((max-min)/10),3),
                round(min+9*((max-min)/10),3),
                round(max,3)),tick = T,cex.axis = .75, mgp = c(2,.7,0))
axis(2, at = y.axis, label = cons_files, las = 1, tick = FALSE, cex.axis =.8)
abline(h = y.axis, lty = 2, lwd = .2, col = "white")
segments(coefs-qnorm(.975)*ses, y.axis+2*adjust, coefs+qnorm(.975)*ses, y.axis+2*adjust, lwd =  1)

segments(coefs-qnorm(.95)*ses, y.axis+2*adjust-.035, coefs-qnorm(.95)*ses, y.axis+2*adjust+.035, lwd = .9)
segments(coefs+qnorm(.95)*ses, y.axis+2*adjust-.035, coefs+qnorm(.95)*ses, y.axis+2*adjust+.035, lwd = .9)
points(coefs, y.axis+2*adjust,pch=21,cex=.8, bg="white")

# Calculate both FRE scores and the Dale-Chall scores
df<- data.frame(texts=cons_text, year=cons_files)

##let's look at FRE
df$read_FRE<-readability(as.character(df$texts), "Flesch")
aggregate(df$read_FRE, by=list(df$year), FUN=mean)

##Dale-Chall measure
df$read_DC<-readability(as.character(df$texts), "Dale.Chall")
aggregate(df$read_DC, by=list(df$year), FUN=mean)

##let's look at all of em

read<-readability(as.character(df$texts))

cor(read$Flesch, read$Dale.Chall)


