Language Models: Auto-Complete
----------------

<p align="justify">
The main goal of this project is to build an auto-complete system. Auto-complete systems are recurrent in the daily use of mobile phones or PCs, for example, every time you google a word in the search bar, you will often find suggestions to help you complete the sentence based on the most frequent searches.
Another example is related to email writing or to text editors, indeed, frequently suggestions are made that could complete the statement.
The underlying focus of this project is therefore to develop an embryonic prototype that combines the skills acquired during the Natural language Processing course.
</p>

<p align="justify">
Language models represent the core of lexical auto-complete system: specifically they associate a probability to a sequence of words, in a way that more "likely" sequences receive higher scores. For example, 
</p>
  
>"I have a pen"
>is expected to have a higher probability than 
>"I am a pen"
>since the first one seems to be a more natural sentence in the real world.

Suppose the user typed 

>"I eat scrambled".

>Therefore it is necessary to find a `x` word such that "I eat scrambled x" receives the highest probability. If x = "eggs", the sentence would be
>"I eat scrambled eggs"

<p align="justify">
  
The model implementation is based on a simple but powerful approach common in machine translation and speech recognition: **N-grams**.
</p>

```python script
import math
import random
import numpy as np
import pandas as pd
import nltk
nltk.data.path.append('.')
```
<p align="justify">
  
The dataset adopted, [twitter data](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip), is a long string that contains a multitude of tweets separated by a break indicator "\n".
</p>


```python script
with open("en_US.twitter.txt", "r") as f:
    data = f.read()
print("Data type:", type(data))
print("Number of letters:", len(data))
print("First 300 letters of the data")
print("-------")
display(data[0:300])
print("-------")

print("Last 300 letters of the data")
print("-------")
display(data[-300:])
print("-------")
```
```
Data type: <class 'str'>
Number of letters: 3335477
First 300 letters of the data
"How are you? Btw thanks for the RT. You gonna be in DC anytime soon? Love to see you. Been way, way too long.\nWhen you meet someone special... you'll know. Your heart will beat more rapidly and you'll smile for no reason.\nthey've decided its more fun if I don't.\nSo Tired D; Played Lazer Tag & Ran A "
```
```
Last 300 letters of the data
"ust had one a few weeks back....hopefully we will be back soon! wish you the best yo\nColombia is with an 'o'...“: We now ship to 4 countries in South America 
(fist pump). Please welcome Columbia to the Stunner Family”\n#GutsiestMovesYouCanMake Giving a cat a bath.\nCoffee after 5 was a TERRIBLE idea.\n"
```

<p align="justify">

Once the dataset is loaded, you proceed with a data split using the "\n" marker as delimiter.You go ahead by removing leading spaces and eliminating the empty strings as shown in the next code box.
</p>
```python script
def split_to_sentences(data):
    """
    Split data by linebreak "\n"
    Args:
    data: str
    Returns:
        A list of sentences
    """
    sentences = data.split("\n")
    # - Remove leading and trailing spaces from each sentence
    # - Drop sentences if they are empty strings.
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0] 
    return sentences    
```

```python script
#Example
x = """
I have a pen.\nI have an apple. \nAh\nApple pen.\n
"""
print(x)
split_to_sentences(x)
```
```
I have a pen.
I have an apple. 
Ah
Apple pen.

['I have a pen.', 'I have an apple.', 'Ah', 'Apple pen.']
```

<p align="justify">
The next step is to tokenize sentences i.e. split a sentence into a list of words. 
At this stage, special attention is also provided to convert all words into lower case so that all words can be treated equally.
</p>

```python script
def tokenize_sentences(sentences):
    # Initialize the list of lists of tokenized sentences
    tokenized_sentences = []
    # Go through each sentence
    for sentence in sentences:
        
        # Convert to lowercase letters
        sentence = sentence.lower()
        
        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)
        
        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)
    
    return tokenized_sentences
  ```
```python script
#Example
sentences = ["Sky is blue.", "Leaves are green.", "Roses are red."]
tokenize_sentences(sentences)
```
```
[['sky', 'is', 'blue', '.'],
 ['leaves', 'are', 'green', '.'],
 ['roses', 'are', 'red', '.']]
```
<p align="justify">
The functions described so far are jointly adopted to apply them to an entire dataset and not only to a single sentence.
</p>


```python script
def get_tokenized_data(data):
    # Get the sentences by splitting up the data
    sentences = split_to_sentences(data)
    # Get the list of lists of tokens by tokenizing the sentences
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences
```
Now the train and the test set are defined, as each sentence is divided into tokens, 

```python script
tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)
train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]
```

```python script
print("{} data are split into {} train and {} test set".format(
    len(tokenized_data), len(train_data), len(test_data)))

print("First training sample:")
print(train_data[0])
      
print("First test sample")
print(test_data[0])
```
```
47961 data are split into 38368 train and 9593 test set
First training sample:
['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the', 'team', 'local', 'company', 'and', 'quality', 'production']
First test sample
['that', 'picture', 'i', 'just', 'seen', 'whoa', 'dere', '!', '!', '>', '>', '>', '>', '>', '>', '>']
```
Due to computational reasons, not all words are used but only the most frequent ones, so it is defined a function that can enumerate the frequency of each word and then consider only those that appear more than N times in the train dataset.
</p>
```python script
def count_words(tokenized_sentences):   
    word_counts = {}
    for sentence in range(len(tokenized_sentences)): 
        # Go through each token in the sentence
        for token in (tokenized_sentences[sentence]): 
            if token not in word_counts.keys(): 
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    return word_counts
```
<p align="justify">
In the definition of auto-complete systems the treatment of words that are missing in the training is of crucial importance. They are known as unknown word or out of vocabulary words. The main related problem is that if they are not observed in training set, the model is incapable of determining which words to suggest.
To handle unknown words during prediction, use a special token to represent all unknown words 'unk'. 
A canonical approach in this context is to modify the training dataset so that it has some 'unknown' words to train on.
In detail, there is a tendency to convert words that occur less frequently into "unk" tokens.
</p>
```python script
def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    # Initialize an empty list to contain the words that
    # appear at least 'minimum_freq' times.
    closed_vocab = []
 
    word_counts = count_words(tokenized_sentences)

    # for each word and its count
    for word, cnt in word_counts.items(): 
        
        # check that the word's count
        # is at least as great as the minimum count
        if cnt>=count_threshold:

            closed_vocab.append(word)
    return closed_vocab
```

```python script
def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
  
    vocabulary = set(vocabulary)
    
    # Initialize a list that will hold the sentences
    # after less frequent words are replaced by the unknown token
    replaced_tokenized_sentences = []
    
    # Go through each sentence
    for sentence in tokenized_sentences:
        
        # Initialize the list that will contain
        # a single sentence with "unknown_token" replacements
        replaced_sentence = []

        # for each token in the sentence
        for token in sentence: 
            
            # Check if the token is in the closed vocabulary
            if token in vocabulary:
                # If so, append the word to the replaced_sentence
                replaced_sentence.append(token)
            else:
                # otherwise, append the unknown token instead
                replaced_sentence.append(unknown_token)

        # Append the list of tokens to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)    
    return replaced_tokenized_sentences
```

The focus now is to jointly use the newly implemented functions in order to identify the less frequent tokens in both training and test sets and then replace them with the "<unk>" marker.

```python script
def preprocess_data(train_data, test_data, count_threshold):
    # Get the closed vocabulary using the train data
    vocabulary = get_words_with_nplus_frequency(train_data,count_threshold)
    train_data_replaced=[]
    test_data_replaced=[]
    # For the train data, replace less common words with "<unk>"
    for sentence in range(len(train_data)):
        parole=[]
        for word in range(len(train_data[sentence])):
            if train_data[sentence][word] in vocabulary:
                parole.append(train_data[sentence][word])
            else:
                parole.append("<unk>")
                
        train_data_replaced.append(parole)
    # For the test data, replace less common words with "<unk>"
    
    for sentence in range(len(test_data)):
        parole_test=[]
        for word in range(len(test_data[sentence])):
            if test_data[sentence][word] in vocabulary:
                parole_test.append(test_data[sentence][word])
            else:
                parole_test.append("<unk>")
        test_data_replaced.append(parole_test)

    return train_data_replaced, test_data_replaced, vocabulary
  ```
The data preprocess is almost finished, it' s only a matter of choosing a minimum frequency for the words.

 ```python script
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data,minimum_freq)
```

 ```python script
print("First preprocessed training sample:")
print(train_data_processed[0])
print()
print("First preprocessed test sample:")
print(test_data_processed[0])
print()
print("First 10 vocabulary:")
print(vocabulary[0:10])
print()
print("Size of vocabulary:", len(vocabulary))
```

```
First preprocessed training sample:
['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the', 'team', 'local', 'company', 'and', 'quality', 'production']

First preprocessed test sample:
['that', 'picture', 'i', 'just', 'seen', 'whoa', 'dere', '!', '!', '>', '>', '>', '>', '>', '>', '>']

First 10 vocabulary:
['i', 'personally', 'would', 'like', 'as', 'our', 'official', 'glove', 'of', 'the']

Size of vocabulary: 14821
```

N-gram based language models
----------------
<p align="justify">
The key assumption behind the model is that the probability of the next word depends exclusively on the previous n-words or n-gram.
The conditional probability for the word at position 't' in the sentence, given that the words preceding it are 
<img src="https://render.githubusercontent.com/render/math?math=w_{t-1}, w_{t-2} \cdots w_{t-n}"> is:
<img src="https://render.githubusercontent.com/render/math?math=P(w_t | w_{t-1}\dots w_{t-n})">. The probability is estimated as follows: <img src="https://render.githubusercontent.com/render/math?math=\hat{P}(w_t | w_{t-1}\dots w_{t-n}) = \frac{C(w_{t-1}\dots w_{t-n}, w_n)}{C(w_{t-1}\dots w_{t-n})}"> where <img src="https://render.githubusercontent.com/render/math?math=C(\cdots)"> denotes the number of occurence of the given sequence. The numerator is the number of times word 't' appears after words t-1 through t-n appear in the training data while the denominator is the number of times word t-1 through t-n appears in the training data.

When computing the counts for n-grams, prepare the sentence beforehand by prepending <img src="https://render.githubusercontent.com/render/math?math=n-1"> starting markers "<s\>" to indicate the beginning of the sentence.
  
>For example, in the bi-gram model (N=2), a sequence with two start tokens "<s\><s\>" should predict the first word of a >sentence.
>So, if the sentence is "I like food", modify it to be "<s\><s\> I like food".
>Also prepare the sentence for counting by appending an end token "<e\>" so that the model can predict when to finish a >sentence.
</p>

The following function count_n_grams computes the n-grams count for each arbitrary n number.
 ```python script
def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):
    # Initialize dictionary of n-grams and their counts
    n_grams = {}
    for sentence in range(len(data)):
        
        # prepend start token n times, and  append <e> one time
        sentences = [start_token]*n+list(data[sentence])+[end_token]
        
        # convert list to tuple
        # So that the sequence of words can be used as
        # a key in the dictionary
        sentences = tuple(sentences)
        
        # Use 'i' to indicate the start of the n-gram
        # from index 0
        # to the last index where the end of the n-gram
        # is within the sentence.
        
        for i in range(0,len(sentences)-n+1): 

            # Get the n-gram from i to i+n
            n_gram = sentences[i:i+n]

            # check if the n-gram is in the dictionary
            if n_gram in n_grams.keys(): 
            
                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1
    return n_grams
 ```
```python script
#Example
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
print("Uni-gram:")
print(count_n_grams(sentences, 1))
print("Bi-gram:")
print(count_n_grams(sentences, 2))
```
```
Uni-gram:
{('<s>',): 2, ('i',): 1, ('like',): 2, ('a',): 2, ('cat',): 2, ('<e>',): 2, ('this',): 1, ('dog',): 1, ('is',): 1}
Bi-gram:
{('<s>', '<s>'): 2, ('<s>', 'i'): 1, ('i', 'like'): 1, ('like', 'a'): 2, ('a', 'cat'): 2, ('cat', '<e>'): 2, ('<s>', 'this'): 1, ('this', 'dog'): 1, ('dog', 'is'): 1, ('is', 'like'): 1}
```
