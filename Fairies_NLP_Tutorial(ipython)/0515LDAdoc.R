############################################
################ R fairies(Adonis Han) #####
# LatentDirichletAllocation ################
# use tm & topicmodel packages(R) to run LDA
#purpose:compare with scikit-learn(python)##
###############Whichone'sbetter#############
############################################

# -Topic modelling provides a quick and convenient way to perform unsupervised classification of a corpus of documents.
# 01 data load ------------------------------------------------------------
library(rJava)
library(tm)
library(topicmodels)
library(readtext)
library(stringr)
library(KoNLP)
library(dplyr)

library(tm)
library(topicmodels)
setwd("C:/Users/Adonishan/r-textmining/corpus")

filenames <- list.files(getwd(),pattern = "*.txt")
files <- lapply(filenames, readLines)
docs <- Corpus(VectorSource(files))


# 02 data pre-processing --------------------------------------------------

# Create the Document-Term Matix
# Formate the files to enable analysis of the terms, then create the Document-Term Matrix which will be analysed in topicmodels 
# package in the next steps

# Remove punctuation - replace puntuation marks with " "
docs <- tm_map(docs, removePunctuation)

# Transform to lower case
docs <- tm_map(docs, content_transformer(tolower))

# Strip digits
docs <- tm_map(docs, removeNumbers)

# Remove Stopwords from standard stopword list
docs <- tm_map(docs, removeWords, stopwords("english"))

# Strip whitespace
docs <- tm_map(docs, stripWhitespace)

# Stem document to ensure words that have same meaning 
# or different verb forms of the same word aren't duplicated
docs <- tm_map(docs, stemDocument)


# remove potentially problematic symbols
toSpace <- content_transformer(function(x, pattern){return (gsub(pattern, " ",x))})
docs <- tm_map(docs, toSpace, "-")
docs <- tm_map(docs, toSpace, "’")
docs <- tm_map(docs, toSpace, "‘")
docs <- tm_map(docs, toSpace, "•")
docs <- tm_map(docs, toSpace, "”")
docs <- tm_map(docs, toSpace, "“")

# fix up 1) differences between us and aussie english
# 2) general errors
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "organiz", replacement = "organ")
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "organis", replacement = "organ")
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "andgovern", replacement = "govern")
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "inenterpris", replacement = "enterpris")
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "team-", replacement = "team")

# stop-word
myStopwords <- c("can", "say","one","way","use",
                 "also","howev","tell","will",
                 "much","need","take","tend","even",
                 "like","particular","rather","said",
                 "get","well","make","ask","come","end",
                 "first","two","help","often","may",
                 "might","see","someth","thing","point",
                 "post","look","right","now","think","‘ve ",
                 "‘re ","anoth","put","set","new","good",
                 "want","sure","kind","larg","yes,","day","etc",
                 "quit","sinc","attempt","lack","seen","awar",
                 "littl","ever","moreov","though","found","abl",
                 "enough","far","earli","away","achiev","draw",
                 "last","never","brief","bit","entir","brief",
                 "great","lot")
docs <- tm_map(docs, removeWords, myStopwords)


# 02-2 Document-term matrix  ----------------------------------------------


# Create document-term matrix
dtm <- DocumentTermMatrix(docs)
rownames(dtm) <- filenames

#collapse matrix by summing over columns
freq <- colSums(as.matrix(dtm))
'''
abstract             accept             access             accord            account              accur 
                 9                 15                 12                 16                  9                  7 
action              activ             actual            acustom            address             affair 
84                 21                 58                  1                 24                  6 
'''
# length should be total number of terms
# how much word are there
length(freq)

# create sort order(descending)
ord <- order(freq, decreasing = TRUE)

# List all terms in decreasing order of freq and write to disk
freq[ord]
'''
           organ            manag             work           system          project          problem           exampl 
             261              220              200              192              183              169              164 
          differ         approach         question            peopl             data          process            chang 
             157              154              152              140              139              137              129 
           model             time              map           import           design            group             issu 
             123              106              105              103               98               96               96 
'''

#save term list
write.csv(freq[ord], "word_freq_0515LDA.csv")

# 03 Load topicmodel and LDA  ---------------------------------------------

# Use Gibbs Sampling on LDA

# Parameters for Gibbs sampling
# 1. set burn in
burnin <- 1000
burnin <- 4000
# 2. set iterations
iter <- 2000
# 3. thin the spaces between samples
thin <- 500
# 4. set random starts at 5
nstart <- 5
# 5. use random integers as seed
seed <- list(254672,109,122887,145629037,2) # set as tutorial 
seed <- list(254542,109,175887,145629037,2)
seed <-list(2003,5,63,100001,765)
# 6. return the highest probability as the result
best <- TRUE
# 7. set number of topics
k <- 5
# 8 .Method
method1 <- c("Gibbs")
method2 <- c("VEM")

# run the LDA model
ldaOut <- LDA(dtm, k, method="Gibbs", control = list(nstart=nstart, seed = seed, best = best, burnin = burnin, iter = iter, thin = thin))
ldaOut <- LDA(dtm, k, method=method2, control = list(nstart=nstart, seed = seed, best = best))


# 04 work through Intelligence  ----------------------------------------------

# view the top 6 terms for each of the 5 topics, create a matrix and write to csv
terms(ldaOut,10)
'''
method = Gibbs Algorithm
      Topic 1    Topic 2     Topic 3    Topic 4   Topic 5   
[1,] "data"     "system"    "said"     "manag"   "question"
[2,] "model"    "one"       "ibi"      "work"    "chang"   
[3,] "exampl"   "process"   "issu"     "project" "like"    
[4,] "approach" "will"      "map"      "consult" "get"     
[5,] "one"      "design"    "knowledg" "risk"    "time"    
[6,] "decis"    "can"       "can"      "team"    "peopl"   
[7,] "view"     "import"    "manag"    "surpris" "thing"   
[8,] "can"      "enterpris" "case"     "flexibl" "well"    
[9,] "way"      "point"     "see"      "organiz" "think"   
[10,] "world"    "develop"   "holm"     "organis" "way"     
'''

'''
method = VEM Algorithm
      Topic 1   Topic 2       Topic 3     Topic 4   Topic 5   
[1,] "work"    "project"     "one"       "model"   "question"
[2,] "one"     "manag"       "system"    "can"     "like"    
[3,] "manag"   "risk"        "design"    "data"    "one"     
[4,] "consult" "uncertainti" "enterpris" "system"  "peopl"   
[5,] "way"     "consequ"     "data"      "one"     "work"    
[6,] "flexibl" "problem"     "point"     "ibi"     "way"     
[7,] "said"    "one"         "use"       "differ"  "get"     
[8,] "surpris" "case"        "peopl"     "use"     "can"     
[9,] "differ"  "state"       "can"       "map"     "map"     
[10,] "see"     "holm"        "problem"   "develop" "thing"   
'''
# write out results
# 1. Docs to Topics
ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics, file=paste("LDAGibbs",k,"DocsToTopics.csv"))
'''
                                                      [,1]
BeyondEntitiesAndRelationships.txt                       2
bigdata.txt                                              1
ConditionsOverCauses.txt                                 5
'''
# 2. top 6 terms in each topic
ldaOut.terms <- as.matrix(terms(ldaOut,6))
write.csv(ldaOut.terms,file=paste("LDAGibbs", k, "TopicsToTerms.csv"))
'''
     Topic 1   Topic 2   Topic 3   Topic 4    Topic 5   
[1,] "issu"    "data"    "project" "question" "organ"   
[2,] "exampl"  "model"   "system"  "time"     "work"    
'''

# 05 Probability calculation ----------------------------------------------

# Find probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities, file=paste('LDAGibbs',k,"TopicProbabilities.csv"))



# investigate topic probabilities data.frame
summary(topicProbabilities)

'''
       V1                V2                V3                V4                V5         
Min.   :0.06583   Min.   :0.04128   Min.   :0.01825   Min.   :0.06244   Min.   :0.05591  
1st Qu.:0.08788   1st Qu.:0.08834   1st Qu.:0.03641   1st Qu.:0.10652   1st Qu.:0.11731  
Median :0.15284   Median :0.11794   Median :0.06700   Median :0.13136   Median :0.17997  
Mean   :0.20743   Mean   :0.18578   Mean   :0.16086   Mean   :0.18400   Mean   :0.26194  
3rd Qu.:0.18961   3rd Qu.:0.19637   3rd Qu.:0.20463   3rd Qu.:0.20310   3rd Qu.:0.42489  
Max.   :0.73806   Max.   :0.64674   Max.   :0.56307   Max.   :0.77307   Max.   :0.72749
'''

# Find relative importance of top 2 topics
# - 1:nrow(dtm) : 1~30
topic1ToTopic2 <- lapply(1:nrow(dtm),function(x)
  sort(topicProbabilities[x,])[k]/sort(topicProbabilities[x,])[k-1])

# Find relative importance of second and thrid most important topics
topic2ToTopics3 <- lapply(1:nrow(dtm), function(x)
  sort(topicProbabilities[x,])[k-1]/sort(topicProbabilities[x,])[k-2])

# write to file
write.csv(topic1ToTopic2, file=paste("LDAGibbs", k, "Topic1ToTopic2.csv"))
write.csv(topic2ToTopics3, file=paste("LDAGibbs", k, "Topic2ToTopic3.csv"))
