# Milestone Report

# Getting Working Directory
getwd()
setwd("./Data Science Capstone")



# Loading All the Libraries

library(tm)
library(NLP)
library(plyr)
library(SnowballC)
library(RWeka)
library(wordcloud)
library(slam)
library(stringi)
library(ggplot2)
library(dplyr)
library(R.utils)
library(openNLP)
library(textmining)


# Loading Raw Datasets
conn <- file("en_US.blogs.txt", open = "rb")
blogs <- readLines(conn, encoding = "UTF-8")
close(conn)

# Read news data in binary mode
conn <- file("en_US.news.txt", open = "rb")
news <- readLines(conn, encoding = "UTF-8")
close(conn)

# Read twitter data in binary mode
conn <- file("en_US.twitter.txt", open = "rb")
twits <- readLines(conn, encoding = "UTF-8")
close(conn)

rm(conn)


## Analyzing Datasets
# Compute words per line info on each line for each data type
rawWPL<-lapply(list(blogs,news,twits),function(x) stri_count_words(x))

# Compute statistics and summary info for each data type
rawstats<-data.frame(
  File=c("blogs","news","twitter"), 
  t(rbind(sapply(list(blogs,news,twits),stri_stats_general),
          TotalWords=sapply(list(blogs,news,twits),stri_stats_latex)[4,])),
  # Compute words per line summary
  WPL=rbind(summary(rawWPL[[1]]),summary(rawWPL[[2]]),summary(rawWPL[[3]]))
)
print(rawstats)

# Sample Data
samplesize <- 20000
set.seed(12345)
twitSample <- twits[rbinom(length(twits)*0.05,length(twits),0.5)]
twitSample <- iconv(twitSample,'UTF-8', 'ASCII', "byte")
newsSample <- news[rbinom(length(news)*0.05,length(news),0.5)]
newsSample <- iconv(newsSample,'UTF-8', 'ASCII', "byte")
blogsSample <- blogs[rbinom(length(blogs)*0.05,length(news),0.5)]
blogsSample <- iconv(blogsSample,'UTF-8', 'ASCII', "byte")

text.Sample <- paste(twitSample,newsSample,blogsSample)

text.Sample <- Corpus(VectorSource(text.Sample))



##  Cleaning the Data
# convert to lower case
text.Sample<- tm_map(text.Sample, tolower)

# remove all punctuations
text.Sample<- tm_map(text.Sample, removePunctuation)

# remove all numbers 
text.Sample<- tm_map(text.Sample, removeNumbers)

## remove whitespace
text.Sample <- tm_map(text.Sample, stripWhitespace)

## force everything back to plaintext document
text.Sample <- tm_map(text.Sample, PlainTextDocument)



# Unigrams
UnigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))

text.Sample.Unigram <- TermDocumentMatrix(text.Sample, control = list(tokenize = UnigramTokenizer))

FreqTerms <- findFreqTerms(text.Sample.Unigram, lowfreq = 4000)

text.Sample.Frequency.Vector.Uni <- sort(rowSums(as.matrix(text.Sample.Unigram[FreqTerms,])),decreasing=TRUE)
text.Sample.Frequency.Dataframe.Uni <- data.frame(word = names(text.Sample.Frequency.Vector.Uni),freq=text.Sample.Frequency.Vector.Uni)

ggplot(data=text.Sample.Frequency.Dataframe.Uni[1:30,],aes(x=word,y=freq)) + 
  geom_bar(stat="identity", fill="orange")  + theme(axis.text.x=element_text(angle=90)) + 
  labs(title="Top 30 Unigrams with a frequency >4000") + 
  labs(x="Unigram") + labs(y="Frequency") 



# Bigrams
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

text.Sample.Bigram <- TermDocumentMatrix(text.Sample, control = list(tokenize = BigramTokenizer))

FreqTerms <- findFreqTerms(text.Sample.Bigram, lowfreq = 600)

text.Sample.Frequency.Vector.Bi <- sort(rowSums(as.matrix(text.Sample.Bigram[FreqTerms,])),decreasing=TRUE)

text.Sample.Frequency.Dataframe.Bi <- data.frame(word = names(text.Sample.Frequency.Vector.Bi),freq=text.Sample.Frequency.Vector.Bi)

ggplot(data=text.Sample.Frequency.Dataframe.Bi[1:30,],aes(x=word,y=freq)) + 
  geom_bar(stat="identity", fill="red") + theme(axis.text.x=element_text(angle=90)) + 
  labs(title="Top 30 Bigrams Frequency >600") + 
  labs(x="Bigrams") + labs(y="Frequency") 



# Trigrams

TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))

text.Sample.Trigram <- TermDocumentMatrix(text.Sample, control = list(tokenize = TrigramTokenizer))

FreqTerms <- findFreqTerms(text.Sample.Trigram, lowfreq = 200)

text.Sample.Frequency.Vector.Tri <- sort(rowSums(as.matrix(text.Sample.Trigram[FreqTerms,])),decreasing=TRUE)
text.Sample.Frequency.Dataframe.Tri <- data.frame(word = names(text.Sample.Frequency.Vector.Tri),freq=text.Sample.Frequency.Vector.Tri)

ggplot(data=text.Sample.Frequency.Dataframe.Tri[1:30,],aes(x=word,y=freq)) + 
  geom_bar(stat="identity", fill="purple") + theme(axis.text.x=element_text(angle=90)) + 
  labs(title="Top 30 Trigrams Frequency > 200") + 
  labs(x="Trigrams") + labs(y="Frequency") 


# Quadgrams

QuadgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 4, max = 4))

text.Sample.Quadgram <- TermDocumentMatrix(text.Sample, control = list(tokenize = QuadgramTokenizer))

FreqTerms <- findFreqTerms(text.Sample.Quadgram, lowfreq = 100)

text.Sample.Frequency.Vector.Quad <- sort(rowSums(as.matrix(text.Sample.Quadgram[FreqTerms,])),decreasing=TRUE)
text.Sample.Frequency.Dataframe.Quad <- data.frame(word = names(text.Sample.Frequency.Vector.Quad),freq=text.Sample.Frequency.Vector.Quad)

ggplot(data=text.Sample.Frequency.Dataframe.Quad[1:30,],aes(x=word,y=freq)) + 
  geom_bar(stat="identity", fill="green") + theme(axis.text.x=element_text(angle=90)) + 
  labs(title="Top 30 Quadgrams Frequency > 100") + 
  labs(x="Quadgrams") + labs(y="Frequency") 



# Wordcloud

wordcloud(words=text.Sample.Frequency.Dataframe.Uni$word,max.words = 300, freq= text.Sample.Frequency.Dataframe.Uni$freq, scale = c(1,1), random.order = F, colors =brewer.pal(20, "Dark2"))
wordcloud(words=text.Sample.Frequency.Dataframe.Bi$word,max.words = 100, freq= text.Sample.Frequency.Dataframe.Bi$freq, scale = c(2,1), random.order = F, colors =brewer.pal(10, "Dark2"))
wordcloud(words=text.Sample.Frequency.Dataframe.Tri$word,max.words = 200, freq= text.Sample.Frequency.Dataframe.Tri$freq, scale = c(3,1), random.order = F, colors =brewer.pal(10, "Dark2"))
