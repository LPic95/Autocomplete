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
</p>
