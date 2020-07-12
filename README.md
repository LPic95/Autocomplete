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


Find tokens that appear at least N times in the training data.
Replace tokens that appear less than N times by <unk>
Note: we omit validation data in this exercise.
In real applications, we should hold a part of data as a validation set and use it to tune our training.
We skip this process for simplicity.




