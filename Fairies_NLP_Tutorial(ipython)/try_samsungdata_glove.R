library(text2vec)
library(Rcampdf)
library(tm)
library(SnowballC)
# data <- read.table(file = "Final_sorted.tsv", sep='\t', header=TRUE)
# data2 <- read.csv(file = "Final_sorted.tsv", sep='\t', header=TRUE)
# d <- readLines('Final_sorted.tsv', skip = 1)

# samsung call data -------------------------------------------------------

#
# 엑셀작업이 빠를듯
# 1) ')' ':' 지우기
# 2) 숫자 지우기
# 3) 첫대문자 소문자로 변환.
#

x <- readLines('Final_sorted.tsv', n = 6)


# 01 pre-text -------------------------------------------------------------

tokens <- space_tokenizer(x)
head(tokens)

# Create iterator over tokens
# vocabulary Terms will be unigram(simple words)
it <- itoken(tokens,progressbar = FALSE)

vocab <- create_vocabulary(it)
head(vocab)
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
head(vocab)

vectorizer <- vocab_vectorizer(vocab)

# skip gram - window size 5 의 context words
# create TCM matrix
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)

# 10차원
glove <- GlobalVectors$new(word_vectors_size = 10 , vocabulary = vocab, x_max = 10)

# word embedding main words
glove_fit <- glove$fit_transform(tcm,n_iter = 10)

# dimension 98개 10차원
dim(glove_fit)
head(glove_fit)

# glove context
glove_context <- glove$components
head(glove_context)

# get word vectors
word_vectors <- glove$get_word_vectors()
word_vectors_2 <- glove_fit + t(glove_context)
head(word_vectors)
head(word_vectors_2)

# save the word vectors
getwd()
write.csv(word_vectors_2, "word_vectors.csv", row.names = TRUE)
write.table(word_vectors_2, "word_vectors.txt", row.names = TRUE)

# try the word samsung
samsung <- word_vectors["google", ,drop=FALSE] - word_vectors["unlock", , drop=FALSE] + word_vectors["call", , drop=FALSE]

samsung <- word_vectors["phone", , drop = FALSE] -
  word_vectors["apple", , drop = FALSE] +
  word_vectors["google", , drop = FALSE]

samsung <- word_vectors_2["phone", , drop = FALSE] -
  word_vectors_2["Samsung", , drop = FALSE] +
  word_vectors_2["google", , drop = FALSE]

cos_sim_samsung <- sim2(word_vectors_2, samsung, method = "cosine", norm = "l2")
cos_sim_samsung[]
head(sort(cos_sim_samsung[,1], decreasing = TRUE), 5)

# Cosine Similarity with word "Samsung"

# \tHHP\tHand       phone     account          6)      google
# 0.6518127   0.6430057   0.5489643   0.5352840   0.5125724



# 02 read again pretrained data -------------------------------------------
# read again
x3 <- read.csv('word_vectors_2.csv',header=FALSE, stringsAsFactors = TRUE)
x3 <- as.vector(x3)

View(x3)
head(x3)
word_vectors_3 <- x3
word_vectors_3 <- data.frame(x3)


x3 <- read.table('word_vectors.csv')

class(x3)
head(x3)
head(word_vectors_3)
View(word_vectors_3)

## try by preprocessed word vector
samsung <- word_vectors_3["phone", , drop = FALSE] -
  word_vectors_3["Samsung", , drop = FALSE] +
  word_vectors_3["google", , drop = FALSE]

cos_sim_samsung <- sim2(word_vectors_3, samsung, method = "cosine", norm = "l2")
cos_sim_samsung[]
head(sort(cos_sim_samsung[,1], decreasing = TRUE), 5)

