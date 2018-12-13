'''
Created by Ping Yuhan on Sep. 18, 2018

This is for NLP which is designed to deal with the predicted alphets and convert them to possible words.

Latest updated on Oct. 22, 2018 

'''

import numpy as np
import time
import ast
import nltk
import random
import pandas as pd
from string import ascii_lowercase
from nltk.corpus import brown
from nltk.corpus import reuters
from collections import deque
from nltk.corpus import words
import re
from collections import Counter
from nltk.corpus import stopwords
from os import path
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt


def words(text):
    return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('holmes.txt').read()))

def P(word, N = sum(WORDS.values())):
    # probability of 'word'
    return WORDS[word] / N

def candidates(word):
    # generate possible spelling corrections for word
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def correction(word):
    # most probable spelling correction for word
    return max(candidates(word), key = P)

def known(words):
    # subset of 'words' that appear in the dictionary of WORDS
    return set(w for w in words if w in WORDS)

def edits1(word):
    #all edits that are one edit away from 'word'
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    # all edits that are two edits away from 'word'
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def spell_check(enhance_list):
    spell_checked_list = []
    if(correct_word == 1):
        spell_checked_list = enhance_list
    else:
        for word in enhance_list:
            spell_checked_list.append(correction(word))
    return spell_checked_list

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

#finds the most probable word from the list of words given by the image
#classifier
correct_word = 0

def probable_word_list(class_input):
    global correct_word
    
    result = {}
    result_correct = []
    print("begining check...")
    
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    
    for word in img_classifier_output:
        if(word in english_vocab):
            correct_word = 1
            result_correct.append(word)
   
    if(correct_word == 0):
        print("finding least distance...")
        for word in img_classifier_output:
            wordlist = set(brown.words())
            i = 0
            minimum_gb = 0
            for actual_word in english_vocab:
                print("finding distance with: ", actual_word)
                minDist = levenshtein(word,actual_word)
                if(i==0):
                    i = 1
                    minimum_gb = minDist
                    result[minimum_gb] = [word]
                elif(minDist < minimum_gb):
                        result.pop(minimum_gb,None)
                        minimum_gb = minDist
                        result[minimum_gb] = [word]
                elif(minDist == minimum_gb):
                        result[minimum_gb].append(word)
    print("Finished Check :)")
    
    if(correct_word == 1):
        return result_correct
    else:
        return result.values()

#Sentiment analysis using NRC emotion lexicon

def sentiment_analysis(word_list):
    
    nrc_lex = pd.read_csv( "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt",sep='\t', names=['word','emotion','association'])
    print ("\n NRC Emotion lexicon loaded...")

    #remove the metadata
    nrc_lex = nrc_lex[21:]
    
    sentiment_enhanced_dict = {}
    sentiment_enhanced_dict['anger'] = []
    sentiment_enhanced_dict['fear'] = []
    sentiment_enhanced_dict['anticipation'] = []
    sentiment_enhanced_dict['trust'] = []
    sentiment_enhanced_dict['surprise'] = []
    sentiment_enhanced_dict['sadness'] = []
    sentiment_enhanced_dict['disgust'] = []
    sentiment_enhanced_dict['joy'] = []
    
    for word in word_list:
        if nrc_lex['word'].str.contains(word).any():

            anger_list = nrc_lex[nrc_lex['word']==word][nrc_lex['emotion']=='anger'][nrc_lex['association'] == 1].index.tolist()
            if len(anger_list) == 1:
                sentiment_enhanced_dict['anger'].append(word)
            fear_list = nrc_lex[nrc_lex['word']==word][nrc_lex['emotion']=='fear'][nrc_lex['association'] == 1].index.tolist()
            if len(fear_list) == 1:
                sentiment_enhanced_dict['fear'].append(word)
            anticipation_list = nrc_lex[nrc_lex['word']==word][nrc_lex['emotion']=='anticipation'][nrc_lex['association'] == 1].index.tolist()
            if len(anticipation_list) == 1:
                sentiment_enhanced_dict['anticipation'].append(word)
            trust_list = nrc_lex[nrc_lex['word']==word][nrc_lex['emotion']=='trust'][nrc_lex['association'] == 1].index.tolist()
            if len(trust_list) == 1:
                sentiment_enhanced_dict['trust'].append(word)
            surprise_list = nrc_lex[nrc_lex['word']==word][nrc_lex['emotion']=='surprise'][nrc_lex['association'] == 1].index.tolist()
            if len(surprise_list) == 1:
                sentiment_enhanced_dict['surprise'].append(word)
            sadness_list = nrc_lex[nrc_lex['word']==word][nrc_lex['emotion']=='sadness'][nrc_lex['association'] == 1].index.tolist()
            if len(sadness_list) == 1:
                sentiment_enhanced_dict['sadness'].append(word)
            joy_list = nrc_lex[nrc_lex['word']==word][nrc_lex['emotion']=='joy'][nrc_lex['association'] == 1].index.tolist()
            if len(joy_list) == 1:
                sentiment_enhanced_dict['joy'].append(word)
            disgust_list = nrc_lex[nrc_lex['word']==word][nrc_lex['emotion']=='disgust'][nrc_lex['association'] == 1].index.tolist()
            if len(disgust_list) == 1:
                sentiment_enhanced_dict['disgust'].append(word)
                
    return sentiment_enhanced_dict

# f = open("wordlist2.txt","r")
f = open('wordlist.txt', 'r')
'''
We still haven't acquired a dataset with images conisiting of paired facial expression and 
hand gestures. So we feed the model with one of the eight emotionsviz: 
anger, fear, anticipation, trust, surprise, sadness, joy, disgust

'''

emotion_received = ['disgust','trust']

img_classifier_output  = f.read().split('\n')
len(img_classifier_output)

enhanced_list = probable_word_list(img_classifier_output)
len(enhanced_list)

#parse it through spell checker
spell_checked_list = spell_check(enhanced_list)


sentiment_enhanced_dict = sentiment_analysis(enhanced_list)

sentiment_enhanced_list = []
for word in emotion_received:
    for word2 in sentiment_enhanced_dict[word]:
        sentiment_enhanced_list.append(word2)



enhanced_list_size = len(enhanced_list)
classifier_output_size = len(img_classifier_output)
sentiment_enhanced_size = len(sentiment_enhanced_list)

print(enhanced_list)
print(sentiment_enhanced_list)

print("Number of outputs generated by the image classifier: ",classifier_output_size)
print("Number of outputs generated by the language enhanced model: ",enhanced_list_size)
print("Number of outputs generated by the sentiment enabled language model: ",sentiment_enhanced_size)



