---
date: 2020-4-20
title: Naive bayes classifier
---

# Assignment 3 Naive bayes classifier
### Hengchao Wang 1001778272
 
 {{% staticref "files/Hengchao_03.ipynb" "newtab" %}}Download my files{{% /staticref %}}

platform: 
I7-9700k GTX-1080ti


```python
import os
import re
import string
import csv
# from bs4 import BeautifulSoup 
import math
import random
from queue import PriorityQueue as PQueue  # priorityQueue
```

## Get data from txt
Some of my thoughts of data pre-processing come from my own homework in my Machine learning class. 


```python
# constant. dictionary of files
train_positive_file_dir = 'aclImdb/train/pos'
train_negitive_file_dir = 'aclImdb/train/neg'

test_positive_file_dir = 'aclImdb/test/pos'
test_negitive_file_dir = 'aclImdb/test/neg'

train_unsup_file_dir = 'aclImdb/train/unsup'
```


```python
def file_name(file_dir):   
    files_name = os.listdir(file_dir)
    return files_name
```


```python
# save filename in txt
def save_name(file_dir, name):
    f = open(name + '.txt' ,'w')  # 'a' add not reset.
    
    files_name = file_name(file_dir)
    for i in files_name:        
        f.write(i)  # string
        f.write("\n")
```


```python
save_name(train_positive_file_dir, 'train_positive_file_dir')
save_name(train_negitive_file_dir, 'train_negitive_file_dir')

save_name(test_positive_file_dir, 'test_positive_file_dir')
save_name(test_negitive_file_dir, 'test_negitive_file_dir')

save_name(train_unsup_file_dir, 'train_unsup_file_dir')
```


```python
# get filename from txt
def get_data(filename):
    f = open(filename +'.txt')
    file_names = []
    for i in f.readlines():
        file_names.append(i.replace("\n", ""))
    return file_names
```


```python
# My thoughts of this part come from my own homework in my Machine learning class. 
# remove some useless charactor at beginning
# load contents into dict
def load_data(fileList, url):
    sentenseList = []
    pa = string.punctuation
    for file in fileList:        
        with open(url +"/"+ file,errors='ignore') as f:
            ori_data = f.read().lower()
            data1 = re.sub('\n{2,6}','  ',ori_data)
            data2 = re.sub('\n',' ',data1)
            data3 = re.sub('  ','yxw ',data2)
            data4 = re.sub("[%s]+"%('"|#|$|%|&|\|(|)|*|+|-|/|<|=|>|@|^|`|{|}|~'), "", data3)
            sentense = re.sub("[%s]+"%('.|?|!|:|;'),'   ',data4)
            sentenseList.append(sentense)
    return sentenseList
```


```python
file_names_train_pos = get_data('train_positive_file_dir')
file_names_train_neg = get_data('train_negitive_file_dir')

file_names_test_pos = get_data('test_positive_file_dir')
file_names_test_neg = get_data('test_negitive_file_dir')

file_names_train_unsup = get_data('train_unsup_file_dir')
```


```python
train_sentenseList1 = load_data(file_names_train_pos, train_positive_file_dir) 
train_sentenseList2 = load_data(file_names_train_neg, train_negitive_file_dir) 

test_sentenseList1 = load_data(file_names_test_pos, test_positive_file_dir) 
test_sentenseList2 = load_data(file_names_test_neg, test_negitive_file_dir) 

train_unsup_sentenseList = load_data(file_names_train_unsup, train_unsup_file_dir) 
```


```python
# merge data
train_target1 = [1]*len(train_sentenseList1)
train_target2 = [0]*len(train_sentenseList2)
train_target = train_target1 + train_target2

train_text1 = train_sentenseList1
train_text2 = train_sentenseList2
train_text = train_text1 + train_text2

test_target1 = [1]*len(test_sentenseList1)
test_target2 = [0]*len(test_sentenseList2)
test_target = test_target1 + test_target2

test_text1 = test_sentenseList1
test_text2 = test_sentenseList2
test_text = test_text1 + test_text2

train_unsup_target = [0]*len(train_unsup_sentenseList)
train_unsup_text = train_unsup_sentenseList
```


```python
train_to_dict = {'content':train_text, 'target':train_target}
test_to_dict = {'content':test_text, 'target':test_target}

train_unsup_to_dict = {'content':train_unsup_text, 'target':train_unsup_target}
```

## Remove stopwords and useless symbol


```python
# Copy the stopwords from wordcloud.
# My thoughts of this part come from my own homework in my Machine learning class. 
stopSet = set({'did', 'such', 'doing', 'down', 'me', 'just', 'very', 'shan', 'against', 't', "you're", 
          'only', "haven't", 'yours', 'you', 'its', 'other', 'we', 'where', 'then', 'they', 'won', "you've",
          'some', 've', 'y', 'each', "you'll", 'them', 'to', 'was', 'once', 'and', 'ain', 'under', 'through',
          'for', "won't", 'mustn', 'a', 'are', 'that', 'at', 'why', 'any', 'nor', 'these', 'yourselves',
          'has', 'here', "needn't", 'm', 'above', 'up', 'more', 'if', 'ma', 'didn', 'whom', 'can', 'have',
          'an', 'should', 'there', 'couldn', 'her', 'how', 'of', 'doesn', "shouldn't", 'further', 
          "wasn't", 'between', 'd', 'wouldn', 'his', 'being', 'do', 'when', 'hasn', "she's", 'by', "should've",
          'into', 'aren', 'weren', 'as', 'needn', 'what', "it's", 'hadn', 'with', 'after', 'he', 'off', 'not',
          'does', 'own', "weren't", "isn't", 'my', 'too', "wouldn't", 'been', 'again', 'same', 'few', "don't",
          'our', 'myself', 'your', 'before', 'about', 'most', 'during', 'll', 'on', 'shouldn', 'is', 'out',
          "shan't", 'below', 'which', 'from', 'she', 'were', 'those', 'over', 'until', 'theirs', 'mightn',
          'yourself', 'i', 'am', 'so', 'himself', 'it', 'had', 'or', 'all', 'while', "aren't", 'ours',
          "that'll", 'but', 'because', 'in', 'now', 'themselves', 'him', "doesn't", 'both', 're', 'wasn',
          's', "hasn't", "didn't", 'their', "mustn't", 'herself', 'the', 'this', 'will', 'isn', "you'd", 
          'haven', 'itself', "couldn't", 'o', 'be', 'don', 'hers', "mightn't", 'having', "hadn't", 'ourselves',
          'who', 'than', 'br'})

# # Remove html characters. I used BeautifulSoup at this part but it's ok to remove this function.
# # So I just put it here as a comment but not use it.

# def strip_html(text):
#     soup = BeautifulSoup(text, "html.parser")
#     return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def remove_stopwords(text, is_lower_case=False):
#     print(text)
    tokens = text.split();
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopSet]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopSet]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#Removing the noisy text
def denoise_text(text):
#     text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    return text
```


```python
# make a copy of data dict
train_to_dict_tmp = {'content':[], 'target':[]}
test_to_dict_tmp = {'content':[], 'target':[]}
unsup_to_dict_tmp = {'content':[], 'target':[]}

for i in range(len(train_to_dict['content'])):
    train_to_dict_tmp['content'].append(denoise_text(train_to_dict['content'][i]))
    train_to_dict_tmp['target'] = train_to_dict['target']

for i in range(len(test_to_dict['content'])):
    test_to_dict_tmp['content'].append(denoise_text(test_to_dict['content'][i]))
    test_to_dict_tmp['target'] = test_to_dict['target']

for i in range(len(train_unsup_to_dict['content'])):
    unsup_to_dict_tmp['content'].append(denoise_text(train_unsup_to_dict['content'][i]))
    unsup_to_dict_tmp['target'] = train_unsup_to_dict['target']

print(test_to_dict_tmp['target'][0])
print(test_to_dict['content'][0])
```

    1
    based on an actual story, john boorman shows the struggle of an american doctor, whose husband and son were murdered and she was continually plagued with her loss    a holiday to burma with her sister seemed like a good idea to get away from it all, but when her passport was stolen in rangoon, she could not leave the country with her sister, and was forced to stay back until she could get i   d    papers from the american embassy    to fill in a day before she could fly out, she took a trip into the countryside with a tour guide    i tried finding something in those stone statues, but nothing stirred in me    i was stone myself    br br suddenly all hell broke loose and she was caught in a political revolt    just when it looked like she had escaped and safely boarded a train, she saw her tour guide get beaten and shot    in a split second she decided to jump from the moving train and try to rescue him, with no thought of herself    continually her life was in danger    br br here is a woman who demonstrated spontaneous, selfless charity, risking her life to save another    patricia arquette is beautiful, and not just to look at    she has a beautiful heart    this is an unforgettable story    br br we are taught that suffering is the one promise that life always keeps   


## Build dictionary


```python
# Define storage of word dictionary and occurrence time
wordSet = []     # store word as a set
word_dictionary = {}  #dictionary, {Integer->index: String->word: }
word_count = {}    #count the frequency of each word, {Integer->index: Integer->frequency}
word_occurrence = {}   # count num of documents containing current word.
word_occurrence_pos = {}    # count num of positive documents containing current word.
word_occurrence_neg = {}    # count num of negative documents containing current word.
```


```python
def get_word_set(wordSet, word_dict):
    for text in word_dict['content']:
        text_words = text.split()
        for word in text_words:
            wordSet.append(word);
    return wordSet
```


```python
wordSet = get_word_set(wordSet, train_to_dict_tmp)
wordSet = get_word_set(wordSet, unsup_to_dict_tmp)
wordSet = set(wordSet)
# wordSet is a set contains all words. Exclude duplicates.
```


```python
# build dictionary
count = 0    # index
for i in wordSet:
    word_dictionary[i] = count
    count = count + 1
```


```python
len(word_dictionary)
```




    199911




```python
# get occurrence time for each word.
def get_count(word_dictionary, word_count, wordSet, word_dict):
    for text in word_dict['content']:
        text_words = text.split()
        for word in text_words:
#             print(word)
            if word in word_dictionary.keys():
                if word_dictionary[word] not in word_count.keys():
                    word_count[word_dictionary[word]] = 1
                else: 
                    word_count[word_dictionary[word]] = word_count[word_dictionary[word]] + 1
    return word_count
```


```python
word_count = get_count(word_dictionary, word_count, wordSet, train_to_dict_tmp)
word_count = get_count(word_dictionary, word_count, wordSet, unsup_to_dict_tmp)
```


```python
# omit rare words if the occurrence is less than five times
for word in wordSet:
    if word_count[word_dictionary[word]] <= 5:
        word_count.pop(word_dictionary[word])
        word_dictionary
        
len(word_count)
```




    47696



## Calculate probability and conditional probability


```python
# cauculate occurrence of each word in pos ang neg. calculate word_occurrence_pos{} and word_occurrence_neg{}
def get_occurrence(word_dict, word_dictionary, wordSet, word_occurrence, word_occurrence_pos, word_occurrence_neg):
    doc_number = 0 
    
    for i in range(len(word_dict['content'])):
        text = word_dict['content'][i]
        target = word_dict['target'][i]
#         print(target)
        doc_number += 1
        text_word_set = set(text.split())
        for word in text_word_set:
            tmp = word_dictionary[word]
            if tmp in word_occurrence.keys(): 
                word_occurrence[tmp] += 1
            else: 
                word_occurrence[tmp] = 1
            if target == 0:
                if tmp in word_occurrence_neg.keys(): 
                   word_occurrence_neg[tmp] += 1
                else: 
                   word_occurrence_neg[tmp] = 1
            if target == 1:
                if tmp in word_occurrence_pos.keys(): 
                   word_occurrence_pos[tmp] += 1
                else: 
                   word_occurrence_pos[tmp] = 1
    return word_occurrence, word_occurrence_pos, word_occurrence_neg, doc_number
```


```python
# get conditional probability with laplace smoothing 
def get_condition_laplace(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data):
    tmp = word_dictionary[word]
    conditional_probability_pos = 0
    conditional_probability_neg = 0
    if tmp in word_occurrence_pos.keys():
        # Laplace Smoothing
        conditional_probability_pos = float(word_occurrence_pos[tmp] + 1)/ float(size_of_data + 2)
    else: conditional_probability_pos = float(1)/ float(size_of_data + 2)
    if tmp in word_occurrence_neg.keys():
        conditional_probability_neg = float(word_occurrence_neg[tmp] + 1)/ float(size_of_data + 2)
    else: conditional_probability_neg = float(1)/ float(size_of_data + 2)
    return conditional_probability_pos, conditional_probability_neg
```


```python
# get conditional probability with m estimate smoothing
def get_condition_m_estimate(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data, m):
    tmp = word_dictionary[word]
    conditional_probability_pos = 0
    conditional_probability_neg = 0
    if tmp in word_occurrence_pos.keys():
        conditional_probability_pos = float(word_occurrence_pos[tmp] + m*(0.5))/ float(size_of_data + m)
    else: conditional_probability_pos = float(m*(0.5))/ float(size_of_data + m)
    if tmp in word_occurrence_neg.keys():
        conditional_probability_neg = float(word_occurrence_neg[tmp] + m*(0.5))/ float(size_of_data + m)
    else: conditional_probability_neg = float(m*(0.5))/ float(size_of_data + m)
    return conditional_probability_pos, conditional_probability_neg
```


```python
# compare conditional_probability
def naive_bayes(text, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data):
    pro_pos = 0
    pro_neg = 0
    for word in text.split():
        if word in word_dictionary.keys():
            tmp = word_dictionary[word]
            if tmp in word_count.keys() and tmp in word_occurrence.keys():
                # laplace
#                 conditional_probability_pos, conditional_probability_neg \
#                 = get_condition_laplace(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data)
                # m estimate, m = 1
#                 conditional_probability_pos, conditional_probability_neg \
#                 = get_condition_m_estimate(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data, 1)
                # m estimate, m = 0.5            
#                 conditional_probability_pos, conditional_probability_neg \
#                 = get_condition_m_estimate(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data, 0.5)
                # m estimate, m = 0.2     
#                 conditional_probability_pos, conditional_probability_neg \
#                 = get_condition_m_estimate(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data, 0.2)
                # m estimate, m = 0.1 
#                 conditional_probability_pos, conditional_probability_neg \
#                 = get_condition_m_estimate(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data, 0.1)
                # m estimate, m = 0.01
                conditional_probability_pos, conditional_probability_neg \
                = get_condition_m_estimate(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data, 0.01)
                pro_pos += math.log(conditional_probability_pos)
                pro_neg += math.log(conditional_probability_neg)
#     print(pro_neg, pro_pos)
    if pro_neg > pro_pos: return 0
    return 1
```


```python
def get_accuracy(word_dict, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg):
    count = 0
    score = 0
    for i in range(len(word_dict['content'])):
        count += 1
        text = word_dict['content'][i]
        target = word_dict['target'][i]
        res = naive_bayes(text, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, len(word_dict['content']))
        if target == res: 
            score += 1
    return float(score)/ float(count) 
```

## Calculate accuracy using dev dataset 
### compare smoothing methods: Laplace and m estimate smoothing.


```python
def fold_validation(word_dict, n_fold):
    dataset_split = []
    text = word_dict['content']
    fold_size = len(text)// n_fold
    index_list = random.sample(range(0,len(text)),len(text))
#     print(index)
    for i in range (n_fold):
        tmp_dict = {}
        tmp_content = []
        tmp_target = []
        for index in index_list[i * fold_size: (i + 1)* fold_size]:
            tmp_content.append(text[index])
            tmp_target.append(word_dict['target'][index])
        tmp_dict = {'content':tmp_content, 'target': tmp_target}
        dataset_split.append(tmp_dict)
    return dataset_split
```


```python
# 5 folds validation
n_fold = 5
data_after_n_fold = fold_validation(train_to_dict_tmp, n_fold)
```


```python
# using development dataset 
# compare m-estimate and laplace smoothing. edit naive_bayes() line 8-13 
score = []
for i in range(n_fold):
    train_dict_final = {}
    for tmp in range(n_fold):
        tmp_content = []
        tmp_target = []
        if tmp != i: 
            tmp_content = tmp_content + data_after_n_fold[tmp]['content']
            tmp_target = tmp_target + data_after_n_fold[tmp]['target']
    train_dict_final = {'content':tmp_content, 'target': tmp_target}
    test_dict_final = data_after_n_fold[i]
    word_occurrence, word_occurrence_pos, word_occurrence_neg, doc_number \
    = get_occurrence(train_dict_final, word_dictionary, wordSet, word_occurrence,\
                  word_occurrence_pos, word_occurrence_neg)
    score.append(get_accuracy(test_dict_final, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg))
print(score)
mean_score = 0
for i in score:
     mean_score += i
print(float(mean_score)/ len(score))
```

    [0.9272, 0.9366, 0.9266, 0.9282, 0.9414]
    0.932


### result of laplace smoothing
[0.9076, 0.9132, 0.9054, 0.9012, 0.9248]
0.91044

### result of m estimate smoothing, m = 1
[0.911, 0.9092, 0.9138, 0.9108, 0.9316]
0.9152799999999999

### result of m estimate smoothing, m = 0.5
[0.9072, 0.9152, 0.9172, 0.9184, 0.9364]
0.91888

### result of m estimate smoothing, m = 0.2
[0.9222, 0.921, 0.921, 0.9176, 0.9244]
0.9212400000000001

### result of m estimate smoothing, m = 0.1
[0.9212, 0.9222, 0.9196, 0.9168, 0.9384]
0.92364

### result of m estimate smoothing, m = 0.01
[0.9272, 0.9366, 0.9266, 0.9282, 0.9414]
0.932

# conclusion: m estimate smoothing have better result. And less m, better result. 

## Derive Top 10 words that predicts positive and negative class


```python
def get_top_10(word_dict, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg):
    top_10_pos = PQueue()  # priorityQueue
    top_10_neg = PQueue()
    for word in word_dictionary.keys():
        if word in word_dictionary.keys():
            tmp = word_dictionary[word]
            if tmp in word_count.keys() and tmp in word_occurrence.keys():
                conditional_probability_pos, conditional_probability_neg \
                = get_condition_laplace(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, len(word_dict['content']))
                top_10_pos.put([conditional_probability_pos * -1, word])
#                 if top_10_pos.qsize() > 10: top_10_pos.get()[-1]
                top_10_neg.put([conditional_probability_neg * -1, word])
#                 if top_10_neg.qsize() > 10: top_10_neg.get()[-1]
    return top_10_pos, top_10_neg
top_10_pos, top_10_neg = get_top_10(train_to_dict ,word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg)


```


```python
print('Top 10 words that predicts positive')
i = 10
while i > 0:
    print(top_10_pos.get(-1)[1])
    i -= 1
print('\n')
print('Top 10 words that predicts negative')
i = 10
while i > 0:
    print(top_10_neg.get(-1)[1])
    i -= 1
```

    Top 10 words that predicts positive
    one
    movie
    film
    like
    good
    time
    great
    see
    story
    well
    
    
    Top 10 words that predicts negative
    movie
    one
    film
    like
    no
    even
    good
    would
    bad
    time


## Using the test dataset calculate the final accuracy.


```python
# final accuracy using m estimate smoothing
word_occurrence, word_occurrence_pos, word_occurrence_neg, doc_number \
    = get_occurrence(train_to_dict_tmp, word_dictionary, wordSet, word_occurrence,\
                  word_occurrence_pos, word_occurrence_neg)
   
score = get_accuracy(train_to_dict_tmp, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg)
score

```




    0.92964



final accuracy using m estimate smoothing = 0.92964
