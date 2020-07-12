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


