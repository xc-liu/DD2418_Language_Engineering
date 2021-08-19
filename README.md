# DD2418_Language_Engineering

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


# Language Engineering



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)


<!-- ABOUT THE PROJECT -->
## About The Project

The projects in this repository are coursework of Language Engineering course in my [master's program in Machine Learning at KTH](https://www.kth.se/en/studies/master/machinelearning/description-1.48533). The projects are divided into 4 assignments:

* Assignment 1: dependency parsing, CKY parsing

This assignment focuses on parsing, analysing the logical syntactic components of sentences. 
In the dependency parsing task, I implemented an algorithm to find the correct moves for a parser, given the parser configuration and the correct final parse tree. 
I also implemented CKY parsing to produce a CKY parse table and then a parse tree from an input sentence.

* Assignment 2: n-gram models and their evaluation, Named Entity Recognition with binary logistic regression, classification for transition-based dependency parsing

In this assignment, I built a bigram model from a given training corpus and evaluated its cross-entropy on the test set.
Binary logistic regression was implemented in the NER task, where the features of the words are not embeddings but hand-picked features. 
Finally, a multinomial logistic regression with early stopping was built to classify the correct parser move. 

* Assignment 3: random Indexing, word2vec, word embedding as features for NER, GloVe

I implemented random index, word2vec, and GloVe methods to extract word embeddings from Harry Potter books. With the embeddings obtained, I redid the NER task with binary logistic regression.

* Assignment 4: gated Recurrent Units, NER with GRUs

Using PyTorch standard mathematical operations, I implemented the GRU method and used the bidirectional GRU to build a NER neural network classifier.
### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [NLTK](https://www.nltk.org/)


