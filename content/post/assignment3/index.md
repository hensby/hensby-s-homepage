---
date: 2020-4-14
title: Assignment3
---
{{% staticref "files/NaiveBayesClassifier.ipynb" "newtab" %}}Download my files{{% /staticref %}}


# Assignment 3 Naive bayes classifier
### Hengchao Wang 1001778272
environment:   
tensorflow 2.2.0  
karas 2.3.1   
sklearn 0.22.1   
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
from queue import PriorityQueue as PQueue
```

## Get data from txt
My thoughts of this part come from my own homework in my Machine learning class. 


```python
# constant
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
def get_data(filename):
    f = open(filename +'.txt')
    file_names = []
    for i in f.readlines():
        file_names.append(i.replace("\n", ""))
    return file_names
```


```python
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
          'who', 'than'})

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
word_occurrence_pos = {}
word_occurrence_neg = {}
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
# wordSet is a set contains all words.
```


```python
# build dictionary
count = 0    # index
for i in wordSet:
    word_dictionary[i] = count
    count = count + 1
word_dictionary
```




    {'fossey': 0,
     'stomachs': 1,
     'overalli': 2,
     'quarter': 3,
     'tearoom': 4,
     'sappier': 5,
     'mcinnerny': 6,
     'mall': 7,
     '0s': 8,
     'portraywhiteonblackviolenceandwinanoscar': 9,
     'honors': 10,
     'ashlee': 11,
     'selfimage': 12,
     'forhire': 13,
     'kamisori': 14,
     'feredes': 15,
     'tbirdspink': 16,
     'mantango': 17,
     'removing': 18,
     'conciliatory': 19,
     'beesslugsrats': 20,
     'vibration': 21,
     'limite': 22,
     'ohsoluckily': 23,
     'limbs': 24,
     'kernoscar': 25,
     'clmence': 26,
     'whiley': 27,
     'brained': 28,
     'nurseslab': 29,
     'husbandplayed': 30,
     'everat': 31,
     'ofir': 32,
     'johnsonand': 33,
     'mysterioussms': 34,
     'bogus': 35,
     'stepdown': 36,
     'licence': 37,
     'irishmen': 38,
     'wussies': 39,
     'swaggart': 40,
     'annakins': 41,
     'ammmmm': 42,
     'colonuswhich': 43,
     'lockpicker': 44,
     'studebakers': 45,
     'imjustheretopickupmypaycheck': 46,
     'hic': 47,
     'norsemen': 48,
     'lovinglyand': 49,
     'yaaden': 50,
     'nephilim': 51,
     'fabritiis': 52,
     'kearn': 53,
     'pernoud': 54,
     'kudoh': 55,
     'wedlock': 56,
     'filmsstop': 57,
     'downeyjr': 58,
     'reubins': 59,
     'absoutley': 60,
     'ecothriller': 61,
     'bulletballets': 62,
     'ageguy': 63,
     'cory': 64,
     'photorealistic': 65,
     'villainies': 66,
     'maartens': 67,
     'misma': 68,
     'gormless': 69,
     'towardsthe': 70,
     'normalize': 71,
     'bossaditya': 72,
     'whoopieee': 73,
     'officeapproved': 74,
     'toods': 75,
     'humanisms': 76,
     'jocket': 77,
     'libidinal': 78,
     'estella': 79,
     'martiono': 80,
     'baseball': 81,
     'hubiriffic': 82,
     'stifling': 83,
     'lsdbecause': 84,
     'shadowswashbuckling': 85,
     'consulate': 86,
     'pact': 87,
     'poorinnocent': 88,
     'mathers': 89,
     'biofilmography': 90,
     'gunilla': 91,
     'rightists': 92,
     'palomas': 93,
     'coupled': 94,
     'interestsympathyadmirationdisgustandhorror': 95,
     'fantasylike': 96,
     'hmmmmmmmmmmmm': 97,
     'coexist': 98,
     'dialoguebut': 99,
     'quartet': 100,
     'swanwhatever': 101,
     'defenders': 102,
     'agenthero': 103,
     'scoyk': 104,
     'neohippies': 105,
     'outherods': 106,
     'androids': 107,
     'tezaab': 108,
     'noah': 109,
     'bassie': 110,
     'selfenclosed': 111,
     'exo': 112,
     'bejing': 113,
     'rewired': 114,
     'barron': 115,
     '1980sforay': 116,
     'trompe': 117,
     'quarry': 118,
     'stephen': 119,
     'caprice': 120,
     'halfmohawks': 121,
     'curatola': 122,
     'tells': 123,
     'psychyatric': 124,
     'farout': 125,
     'mewing': 126,
     'alcoholdrugs': 127,
     'dargaha': 128,
     'tattoine': 129,
     'harmlesslooking': 130,
     'sobels': 131,
     'tehanu': 132,
     'edmund': 133,
     'fairly': 134,
     'nonzombified': 135,
     'cocacolas': 136,
     'hilles': 137,
     'lacanlike': 138,
     'colorfullypainted': 139,
     'pottymouth': 140,
     'handsomeshes': 141,
     'likesteve': 142,
     'mdm': 143,
     'walton': 144,
     'grandparent': 145,
     'ophls': 146,
     'fore': 147,
     'biology': 148,
     'outshadowed': 149,
     'muchballyhooed': 150,
     'generalin': 151,
     'goreheads': 152,
     'janningss': 153,
     'sens': 154,
     'manwith': 155,
     'sujamal': 156,
     'nailbitingly': 157,
     'shandling': 158,
     'sukima': 159,
     'uncontained': 160,
     'nonwitness': 161,
     'fatherson': 162,
     'holding': 163,
     'heretofore': 164,
     'raisins': 165,
     'devorah': 166,
     'gutbased': 167,
     'rendevous': 168,
     'breezy': 169,
     'fijians': 170,
     'jehadi': 171,
     'audiencegoers': 172,
     'listthere': 173,
     'darkfor': 174,
     'affleck': 175,
     'thrumming': 176,
     'pattie': 177,
     'longlimbed': 178,
     'abcs': 179,
     'unthreatened': 180,
     'perceptions': 181,
     'whittle': 182,
     'latjo': 183,
     'raftfirst': 184,
     'bootblack': 185,
     'houseother': 186,
     'ural': 187,
     'culthorror': 188,
     'ungraceful': 189,
     'murakamiare': 190,
     'carnaval': 191,
     'anglominiseries': 192,
     'stupidvery': 193,
     'pixelbypixel': 194,
     'bist': 195,
     'waythe': 196,
     'doody': 197,
     'bequeathed': 198,
     'derric': 199,
     'stockholmers': 200,
     'falconlike': 201,
     'dechifered': 202,
     '2009ii': 203,
     'wittywhich': 204,
     'forgivingly': 205,
     'espouseeven': 206,
     'outofrole': 207,
     'tronish': 208,
     'interpool': 209,
     'genzel': 210,
     'untitled': 211,
     'suderland': 212,
     'instill': 213,
     'trackless': 214,
     'f____g': 215,
     'nietos': 216,
     'deliriousness': 217,
     'maggi': 218,
     'sinuously': 219,
     'inaproppriately': 220,
     'yeaaaaaaaaaaaaah': 221,
     'secondarystarmanuntouchables': 222,
     'c64': 223,
     'kisser': 224,
     'karrina': 225,
     'weeeeee': 226,
     '80th': 227,
     'zombiesno': 228,
     'tabool': 229,
     'haver': 230,
     'ocp': 231,
     'incandescent': 232,
     'satrapibr': 233,
     'bhula': 234,
     'askbr': 235,
     'goldblume': 236,
     'darkstorm': 237,
     'consadines': 238,
     'ncaa': 239,
     'averydawn': 240,
     'dialectical': 241,
     'tinged': 242,
     'frolically': 243,
     'capacityhes': 244,
     'portholes': 245,
     'allowed': 246,
     'arakis': 247,
     'anoints': 248,
     'firmincluding': 249,
     'saleen': 250,
     'helping': 251,
     'mtv2': 252,
     'sceneculture': 253,
     'youssou': 254,
     'danon': 255,
     'apprently': 256,
     'curis': 257,
     'longanticipated': 258,
     'believale': 259,
     'firecrackers': 260,
     'lifetimes': 261,
     'highkick': 262,
     'selfmutilation': 263,
     'doinks': 264,
     'pseudodisaffected': 265,
     'nikalta': 266,
     'nearnothing': 267,
     'filmsomewhat': 268,
     'plummage': 269,
     'jitterbug': 270,
     'fkmag': 271,
     'boygirlmom': 272,
     'valuebr': 273,
     'boardsim': 274,
     'extricates': 275,
     'capitalizes': 276,
     'weinzweig': 277,
     'hermano': 278,
     'jerilderie': 279,
     'hollowzombieevileyes': 280,
     'handpuppet': 281,
     'papercutout': 282,
     'sagamore': 283,
     'afa': 284,
     'brillianceive': 285,
     'togetherthings': 286,
     'forme': 287,
     'dermatonecrotic': 288,
     'scifiish': 289,
     'ghabah': 290,
     'swiftheart': 291,
     'garagebands': 292,
     'basha': 293,
     'towed': 294,
     'ouzts': 295,
     'refugeesbut': 296,
     'dynamites': 297,
     '7though': 298,
     'darkwolfs': 299,
     'varricks': 300,
     '101': 301,
     'wintons': 302,
     'hypochondriac': 303,
     'arablooking': 304,
     'johnerik': 305,
     'regifted': 306,
     'guysbr': 307,
     'changeover': 308,
     'reavis': 309,
     'thundered': 310,
     'unreasonable': 311,
     'lengles': 312,
     'goodhandsome': 313,
     'takinggiving': 314,
     'mcphail': 315,
     'mindwarping': 316,
     'infective': 317,
     'wittering': 318,
     'antipc': 319,
     'kilo': 320,
     'copulating': 321,
     'verheyen': 322,
     'icetwho': 323,
     'illnamed': 324,
     '1985id': 325,
     'gastone': 326,
     'pence': 327,
     'beforei': 328,
     'flamengos': 329,
     'hisaishis': 330,
     'snowmen': 331,
     'exemplar': 332,
     'toappreciatethebeautyofthe': 333,
     'galego': 334,
     'moffatt': 335,
     'plumpish': 336,
     'folcanet': 337,
     'promptly': 338,
     'fasso': 339,
     '1500000': 340,
     'artistentrepreneur': 341,
     'lamarrwho': 342,
     'rextasy': 343,
     'wanly': 344,
     'nostalgic': 345,
     'stentorian': 346,
     'penny': 347,
     'onenough': 348,
     'thingno': 349,
     'tents': 350,
     'speaches': 351,
     'trueblood': 352,
     'fieldtrip': 353,
     'ranald': 354,
     'belowaverage': 355,
     'chemicals': 356,
     'radioactively': 357,
     'chasetype': 358,
     'bussized': 359,
     'fob': 360,
     'shermeyer': 361,
     'ona': 362,
     'sayis': 363,
     'memoriam': 364,
     'dermont': 365,
     'gialli': 366,
     'tendentious': 367,
     'chunksoclich': 368,
     'eurasian': 369,
     'ornithochirus': 370,
     'pantheresquire': 371,
     'extravagant': 372,
     'thistake': 373,
     'somersaulted': 374,
     'cannonballs': 375,
     'pendelton': 376,
     'oldlady': 377,
     'bangor': 378,
     'alienness': 379,
     'cheatedupon': 380,
     'taverneir': 381,
     'deadalive': 382,
     'trickortreating': 383,
     'overgrowth': 384,
     'fleissespecially': 385,
     'sisterthere': 386,
     'haase': 387,
     'dressact': 388,
     'privatizing': 389,
     '76minute': 390,
     'neoaunatural': 391,
     'corruptbourgeois': 392,
     'aesops': 393,
     'technologyinfesting': 394,
     'laughin': 395,
     'americanese': 396,
     'connotation': 397,
     'breziner': 398,
     'meim': 399,
     'valverde': 400,
     'whackos': 401,
     'yakuzalike': 402,
     'danger': 403,
     'malemale': 404,
     'biographies': 405,
     'bengal': 406,
     'valencia': 407,
     'despots': 408,
     'milinkovic': 409,
     'nadiadwala': 410,
     'longonow': 411,
     'hendel': 412,
     'strangenerdy': 413,
     '60minute': 414,
     'buseyand': 415,
     'halfspeed': 416,
     'dadoo': 417,
     'permating': 418,
     'timeshift': 419,
     'couscous': 420,
     'funcampycultmovie': 421,
     'boombox': 422,
     'inyourface': 423,
     'hauer': 424,
     'motorhomes': 425,
     '12yrold': 426,
     'raised': 427,
     'dreamall': 428,
     'localif': 429,
     'meatless': 430,
     'cocainesnorting': 431,
     'emeraldas': 432,
     'incapacitated': 433,
     'facsimily': 434,
     'cannibus': 435,
     'gammeras': 436,
     'asproon': 437,
     'cleancut': 438,
     'tinnitus': 439,
     'scribbly': 440,
     'ali': 441,
     'knottingup': 442,
     'jumpout': 443,
     'personalitiespolitics': 444,
     'distilling': 445,
     'turbanstyle': 446,
     'wanted': 447,
     'prvertbut': 448,
     'galreel': 449,
     'lucio': 450,
     '19945': 451,
     'strictness': 452,
     'lip': 453,
     'accentuation': 454,
     'tumacacori': 455,
     'crate': 456,
     'ninetiesera': 457,
     'alexies': 458,
     'tuetonic': 459,
     'muddledness': 460,
     'objectprop': 461,
     '4\\10': 462,
     'smilebr': 463,
     'lumphammer': 464,
     'alienqueen': 465,
     'babyflesh': 466,
     'film16mmdvdvhs': 467,
     'storyhe': 468,
     'teenskids': 469,
     'parlous': 470,
     'flagwaving': 471,
     'retaliationthe': 472,
     'superbeing': 473,
     'cappelletti': 474,
     'boundlike': 475,
     'earning': 476,
     'weatherbeaten': 477,
     'comtitlett1445990usercomments1br': 478,
     'passiveseeing': 479,
     'doughnuts': 480,
     'animatedpuppet': 481,
     'pussies': 482,
     'stillread': 483,
     'formerpunkrockerturnedorchestralarranger': 484,
     'catchmeifcan': 485,
     'resurecting': 486,
     'watchthistype': 487,
     'mailshirts': 488,
     'wui': 489,
     'therejust': 490,
     'exonerates': 491,
     'reached': 492,
     'imodium': 493,
     'podeswa': 494,
     'alongcreating': 495,
     'vday': 496,
     'superprotestant': 497,
     'horseshack': 498,
     'tokuichi': 499,
     'geniussquallbecomes': 500,
     'kareeenas': 501,
     'veillance': 502,
     'siege': 503,
     'tractacus': 504,
     'pachebels': 505,
     'scumbag': 506,
     'regentreleasing': 507,
     'excitingbut': 508,
     'separatist': 509,
     'lowcountry': 510,
     'lykawa': 511,
     'ghandi': 512,
     'incomprehension': 513,
     'howler': 514,
     'medhi': 515,
     'fifes': 516,
     'milchs': 517,
     'disfigures': 518,
     'birma': 519,
     'thinker1691': 520,
     'erdogan': 521,
     'necktie': 522,
     'vitos': 523,
     'accounted': 524,
     'lea': 525,
     'soxangels': 526,
     'inseparable': 527,
     'september': 528,
     '1790sto': 529,
     'genoa': 530,
     'mildewing': 531,
     'raineys': 532,
     'adultchild': 533,
     'completeor': 534,
     'haverchuck': 535,
     'dashiell': 536,
     'audiocassette': 537,
     'spaciba888hotmail': 538,
     'abscbn': 539,
     'sotospeak': 540,
     'munrovehicle': 541,
     'offroad': 542,
     'hellolarry': 543,
     'rivalingthat': 544,
     'leathergoods': 545,
     'mr': 546,
     'deathimpending': 547,
     'gutenbergs': 548,
     'trruck': 549,
     'splenda': 550,
     'selfrightousness': 551,
     'fave': 552,
     'theye': 553,
     'aahed': 554,
     'affronts': 555,
     'oshn': 556,
     'footsoldier': 557,
     'loungecore': 558,
     'stadlen': 559,
     'universalappeal': 560,
     'topactor': 561,
     '30000': 562,
     'walon': 563,
     'einarsson': 564,
     'helio': 565,
     'wide': 566,
     'pentecost': 567,
     'zoetrope': 568,
     'couldint': 569,
     'bannerlilith': 570,
     'helmond': 571,
     'bugwith': 572,
     'voltage': 573,
     'eweekly': 574,
     'sceneroom': 575,
     'broadcasting': 576,
     'disapprovement': 577,
     'confusedwhich': 578,
     'nonsequel': 579,
     'arpeggiating': 580,
     'muchlamented': 581,
     'mulgrew': 582,
     'nonsequituurs': 583,
     'hillbilles': 584,
     'nonactive': 585,
     'shipwrecks': 586,
     'videoclips': 587,
     'phsyco': 588,
     'bedder': 589,
     'cutscenes': 590,
     'fuss': 591,
     'bluer': 592,
     'dunway': 593,
     '1665': 594,
     'parasitical': 595,
     'chainstores': 596,
     'deborah': 597,
     'manchuria': 598,
     'dethaw': 599,
     'metalhard': 600,
     'registersbut': 601,
     'unscrupulously': 602,
     'culturaloriented': 603,
     'dreichness': 604,
     'dictature': 605,
     'altos': 606,
     'spainish': 607,
     'voyages': 608,
     'shipyards': 609,
     'singeractor': 610,
     'pigtails': 611,
     'jena': 612,
     'moonless': 613,
     'passportvisas': 614,
     'shackdwelling': 615,
     'durangomexico': 616,
     'texan': 617,
     'asylm': 618,
     'eg': 619,
     'badnoplot': 620,
     'reconnects': 621,
     'slinky': 622,
     'multipicture': 623,
     'shocktactics': 624,
     'elementhe': 625,
     'sips': 626,
     'indolently': 627,
     'mayhem': 628,
     'mused': 629,
     'naya': 630,
     'ruddock': 631,
     'zag': 632,
     'harrisburg': 633,
     'natgeo': 634,
     'samples': 635,
     'vcnvrmzx2kms': 636,
     'silencer': 637,
     'vigorously': 638,
     'hogbottoms': 639,
     'depot': 640,
     'psychotherapists': 641,
     'bilals': 642,
     'saviour': 643,
     'molieres': 644,
     'robinsbraces': 645,
     'significiant': 646,
     'overlypadded': 647,
     'boywell': 648,
     'bser': 649,
     'goblinmeal': 650,
     'adone': 651,
     'bodyhopping': 652,
     'grdos': 653,
     'travelish': 654,
     'ohtar': 655,
     'mediacrew': 656,
     'bunyan': 657,
     'sirks': 658,
     'rueff': 659,
     'mapes': 660,
     'rhinohating': 661,
     'recuperates': 662,
     'recordsetting': 663,
     'writerdirectorproducereditorstar': 664,
     'backdancers': 665,
     'japaneseamericans': 666,
     'sunya': 667,
     'brontes': 668,
     'mcgarrigle': 669,
     'cheapscare': 670,
     'egyptianlike': 671,
     'admonitions': 672,
     'exacerbates': 673,
     '1986but': 674,
     '_want_': 675,
     'cautionary': 676,
     'realllllllly': 677,
     'hoc': 678,
     'overthecounter': 679,
     'chairis': 680,
     'authoritarianism': 681,
     'larkspur': 682,
     'unsuspensful': 683,
     'high8': 684,
     'bighouseazyahoo': 685,
     'todayrachel': 686,
     'calchas': 687,
     'bartilson': 688,
     'prefaced': 689,
     'trays': 690,
     'stanislavskys': 691,
     'walas': 692,
     'regressivecollective': 693,
     'cartooncomics': 694,
     'maki': 695,
     '_apocalyptically': 696,
     'hanksryan': 697,
     'sdoa': 698,
     'augustaless': 699,
     'centering': 700,
     'exscenesters': 701,
     'dwarfswho': 702,
     'horrorfeast': 703,
     'upakhyan': 704,
     'mauryan': 705,
     'nidia': 706,
     'holograms': 707,
     'wisepaul': 708,
     'mockudramas': 709,
     'tinsel': 710,
     'sorcia': 711,
     'oblix': 712,
     'togarmary': 713,
     'nazisrapemysterious': 714,
     'livresse': 715,
     'socialhistorical': 716,
     'bering': 717,
     'bros': 718,
     'edtv': 719,
     'verne': 720,
     'tyold': 721,
     'mohile': 722,
     'golsmith': 723,
     'israeliss': 724,
     'robbed': 725,
     'ning': 726,
     'halfjoking': 727,
     'gundammech': 728,
     'conlans': 729,
     'vast': 730,
     'keyshia': 731,
     'panpeter': 732,
     'pacedfilled': 733,
     'stillfaithful': 734,
     'motivesid': 735,
     'effectas': 736,
     'holcroftbody': 737,
     'werewolfhorror': 738,
     'bondspy': 739,
     'dialogueactingstory': 740,
     'pressure': 741,
     'lena': 742,
     'eavesdropping': 743,
     'storesteal': 744,
     'rescueturning': 745,
     'rere': 746,
     'sentiments': 747,
     'kiddish': 748,
     'wilks': 749,
     'mamiyas': 750,
     'parka': 751,
     '1970a': 752,
     'stepdaughter': 753,
     'sluggers': 754,
     'bestnaseerdun': 755,
     'prejudged': 756,
     'directingfailures': 757,
     'guttenbergis': 758,
     'explicityet': 759,
     'rob': 760,
     'brims': 761,
     'wellaccording': 762,
     'beatific': 763,
     'thirdtolast': 764,
     'saulnier': 765,
     'euroanimation': 766,
     'aremr': 767,
     'awile': 768,
     'plasticboob': 769,
     'resultthis': 770,
     'delapaz': 771,
     'idealists': 772,
     'returning': 773,
     'pr': 774,
     'reconnecting': 775,
     '2006': 776,
     'kush': 777,
     '1930smodern': 778,
     'iris_2_youyahoo': 779,
     'beforetime': 780,
     'insisting': 781,
     'vase': 782,
     'responsibilty': 783,
     'brusilov': 784,
     'occupies': 785,
     'darby': 786,
     'lovinglymade': 787,
     'mitsuo': 788,
     'leeveeo': 789,
     'dazzle': 790,
     'elia': 791,
     'lycos': 792,
     'peoplenot': 793,
     'tudman': 794,
     'hijinks': 795,
     'reinsert': 796,
     'hadi': 797,
     'kainen': 798,
     'aylesworth': 799,
     'ilene': 800,
     'flaxenhaired': 801,
     'miniaturist': 802,
     'wyoming': 803,
     'feared': 804,
     'anticlimatic': 805,
     'opinionthere': 806,
     'compartments': 807,
     'bogaert': 808,
     'almoust': 809,
     'gerhardt': 810,
     'pivitol': 811,
     'reaperjohn': 812,
     'degeneratesserial': 813,
     'joanas': 814,
     'roedean': 815,
     'wifebeating': 816,
     'searchlights': 817,
     'stag': 818,
     'knowsunless': 819,
     'talladega': 820,
     'chattier': 821,
     'megas': 822,
     'nonstereotypical': 823,
     'severisons': 824,
     'battleground': 825,
     'tomlin': 826,
     'philosophicalbullshit': 827,
     'programfinally': 828,
     'sonformer': 829,
     'mains': 830,
     'cowcreature': 831,
     'moviethen': 832,
     'offyou': 833,
     'connellylipstick76': 834,
     'dianabob': 835,
     'darnells': 836,
     'ericgeorge': 837,
     'mourir': 838,
     'breast': 839,
     'sebastians': 840,
     'ciggylightingbyfingertip': 841,
     'cameraperson': 842,
     'crains': 843,
     'melon': 844,
     'deterioration': 845,
     'hockeyplaying': 846,
     'aliases': 847,
     'bopheck': 848,
     '5075': 849,
     'ackroyds': 850,
     'posative': 851,
     'compatriots': 852,
     'payoffdream': 853,
     'unsubdued': 854,
     'kovacic': 855,
     'degree': 856,
     'falangists': 857,
     'credibly': 858,
     'brownnovel': 859,
     'cordelier': 860,
     'fortissimo': 861,
     'pseudosci': 862,
     'beavertrapper': 863,
     'rz': 864,
     'perplexing': 865,
     'alwyn': 866,
     'whitepower': 867,
     'progressional': 868,
     'givenbr': 869,
     'decampitated': 870,
     'obadai': 871,
     'talkers': 872,
     'writernarrator': 873,
     'beachy': 874,
     'alphawoman': 875,
     'slowlytimed': 876,
     'charnier': 877,
     'carraways': 878,
     'refamiliarise': 879,
     'obidient': 880,
     'cyberscum': 881,
     'kissingintheswimmingpool': 882,
     'gadar': 883,
     '_toy': 884,
     'warriorpossibly': 885,
     'uds': 886,
     'haute': 887,
     'rulin': 888,
     'tomiche': 889,
     'cringeathon': 890,
     'waspworthy': 891,
     'demonstrating': 892,
     'dedlock': 893,
     'valdezs': 894,
     'indiana': 895,
     'horribletheres': 896,
     'novels': 897,
     'doubts': 898,
     'audiencescaring': 899,
     'submichael': 900,
     'wells': 901,
     'anabelle': 902,
     'convenientlynearby': 903,
     'sixthsense': 904,
     'mister': 905,
     'insidei': 906,
     'soundit': 907,
     'overrationalized': 908,
     'videozone': 909,
     'ultracatholic': 910,
     '091005': 911,
     'ransohoff': 912,
     'unites': 913,
     'bloppers': 914,
     'jimunji': 915,
     'federations': 916,
     '99yearsold': 917,
     'overdramaticizing': 918,
     'grandkids': 919,
     'agility': 920,
     'spygenre': 921,
     'ludlums': 922,
     'cliffface': 923,
     '20searly': 924,
     'sanitarium': 925,
     'kolross': 926,
     'computerespionage': 927,
     'loosecannon': 928,
     'brucethough': 929,
     'tards': 930,
     'burgade': 931,
     'bps': 932,
     'tracee': 933,
     'nonflashback': 934,
     'renfroreunite': 935,
     'harmonizers': 936,
     'bobe': 937,
     'dashon': 938,
     'jarols': 939,
     'defilers': 940,
     'hmmmnotice': 941,
     'rdt': 942,
     'antigem': 943,
     'repenting': 944,
     'weaken': 945,
     'mogosoaia': 946,
     'limitation': 947,
     'onedimensionality': 948,
     'galsworthy': 949,
     'genredistilling': 950,
     'catmouse': 951,
     'exverger': 952,
     'rhoades': 953,
     'rightsbut': 954,
     'mitchel': 955,
     'praisebr': 956,
     'allaudience': 957,
     'electorate': 958,
     'plotmore': 959,
     'frustratingly': 960,
     'whope': 961,
     'referenceladen': 962,
     'glendenning': 963,
     'sdds': 964,
     'arranged': 965,
     'citations': 966,
     'modernlife': 967,
     'fredas': 968,
     'clamps': 969,
     'keyboardists': 970,
     'aptstenements': 971,
     'onde': 972,
     'mannerismsgait': 973,
     'hives': 974,
     'giallio': 975,
     'toho': 976,
     'yemen': 977,
     'combaron': 978,
     'complementary': 979,
     'whipsmart': 980,
     'descends': 981,
     'abhays': 982,
     'bejebees': 983,
     'performedbut': 984,
     'hotelbr': 985,
     'witht': 986,
     'emiles': 987,
     'sturla': 988,
     'applesthis': 989,
     'brouhaha': 990,
     'mandalys': 991,
     'victory': 992,
     'reformatory': 993,
     'ops': 994,
     'scorcher': 995,
     'trailerbut': 996,
     'turpitude': 997,
     'escapeaccidentally': 998,
     'wrongdoings': 999,
     ...}




```python
len(word_dictionary)
```




    199912




```python

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
# omit rare words for example if the occurrence is less than five times
for word in wordSet:
    if word_count[word_dictionary[word]] <= 5:
        word_count.pop(word_dictionary[word])
        word_dictionary
        
len(word_count)
```




    47697




```python
word_count
```




    {20283: 128732,
     90926: 9451,
     138601: 37603,
     141385: 1423,
     139983: 7870,
     76262: 12170,
     12801: 1976,
     82596: 268,
     97587: 348,
     99782: 1014,
     174208: 2181,
     64332: 1933,
     143204: 17,
     49832: 12752,
     96806: 12974,
     22460: 143,
     156352: 31,
     107627: 16174,
     23145: 31,
     154730: 6,
     38087: 20145,
     162290: 1076,
     79378: 721,
     59630: 20549,
     127664: 881,
     68744: 19,
     49881: 139,
     35610: 851,
     138983: 3252,
     28917: 123,
     88824: 1457,
     13643: 10270,
     111191: 1506,
     22989: 2158,
     76306: 1140,
     29119: 625,
     121999: 42,
     103467: 93,
     29303: 6420,
     118562: 155,
     137557: 6,
     116752: 1372,
     1063: 10879,
     107271: 2879,
     157345: 140,
     65096: 669,
     179046: 11,
     186976: 876,
     24028: 40,
     136427: 1791,
     102044: 423,
     164644: 47,
     176722: 28,
     96743: 28,
     96080: 3483,
     47484: 15762,
     11823: 54,
     198936: 29453,
     116691: 796,
     158758: 1264,
     9799: 8936,
     132375: 4544,
     597: 95,
     10218: 1249,
     198258: 7,
     187076: 249,
     178189: 1540,
     63725: 1746,
     184973: 242,
     129861: 216,
     177412: 23,
     75232: 114,
     122618: 740,
     134335: 640,
     70168: 4483,
     135922: 855,
     166174: 11175,
     46251: 34858,
     109012: 5248,
     193161: 141,
     4051: 281,
     32708: 3132,
     95225: 86,
     126530: 355,
     133471: 89,
     158568: 133,
     148894: 92,
     115305: 5274,
     129894: 172,
     54701: 606,
     107273: 311,
     51518: 28,
     175217: 125,
     123522: 97,
     88442: 61,
     114556: 27239,
     180622: 7945,
     165573: 42,
     62406: 1459,
     144208: 2298,
     195118: 92,
     114867: 1795,
     87285: 707,
     23335: 4277,
     140322: 42,
     112080: 256,
     82577: 7,
     173635: 93,
     26606: 38,
     112776: 363,
     118430: 4841,
     177262: 168,
     157905: 2276,
     8749: 11569,
     159093: 3427,
     141055: 629,
     40608: 26,
     107726: 38,
     171255: 99,
     136940: 7091,
     104906: 500,
     101013: 1488,
     72096: 1507,
     161832: 101,
     195292: 115922,
     26357: 243,
     85881: 72,
     22022: 4794,
     147374: 3577,
     169224: 11930,
     174259: 13430,
     10771: 36274,
     85157: 23654,
     18016: 14921,
     181483: 23771,
     5495: 34118,
     36780: 13,
     21673: 4778,
     65456: 4024,
     129972: 17,
     91723: 611,
     126424: 275,
     99085: 160,
     44199: 118,
     110396: 22,
     147443: 2836,
     35243: 3311,
     128380: 2048,
     31805: 2932,
     170081: 2277,
     32121: 144,
     21202: 2619,
     159343: 39,
     39887: 5586,
     2686: 4292,
     177201: 59415,
     160523: 6649,
     114700: 360,
     83873: 5395,
     105851: 6,
     140300: 18,
     189557: 301420,
     85385: 2506,
     79028: 829,
     140286: 248,
     73041: 6512,
     34396: 840,
     150941: 99,
     78005: 100,
     87892: 3760,
     70733: 531,
     164700: 34699,
     173732: 2240,
     32926: 12861,
     139473: 3698,
     32173: 3272,
     155381: 186,
     191855: 52,
     71410: 1188,
     173212: 23,
     120630: 1589,
     133383: 14,
     67259: 7585,
     94388: 5528,
     177606: 19392,
     67786: 2424,
     17643: 7485,
     119799: 114,
     123662: 910,
     41081: 50,
     88431: 3093,
     15515: 743,
     184517: 1212,
     94699: 36,
     183713: 198,
     68176: 6,
     94961: 4709,
     97231: 11,
     114018: 82,
     73959: 34,
     35144: 5950,
     74121: 1013,
     156681: 29,
     123515: 37,
     129496: 3728,
     22123: 5839,
     142974: 75,
     60285: 1368,
     43747: 44,
     138689: 176,
     54757: 1390,
     113584: 578,
     171996: 1090,
     138121: 32,
     45736: 4187,
     79232: 598,
     103378: 62,
     74107: 420,
     47186: 163,
     181467: 8,
     192517: 740,
     37798: 6872,
     28426: 3004,
     177533: 188,
     64377: 190,
     199414: 36299,
     177219: 523,
     32852: 107,
     107855: 8,
     81756: 1295,
     55587: 126,
     87823: 68,
     43051: 6443,
     84445: 85,
     30374: 1877,
     151854: 34,
     58430: 14,
     195257: 280,
     66024: 27,
     136527: 67,
     13673: 76,
     93861: 170,
     123251: 578,
     49966: 294,
     78900: 4714,
     11266: 109,
     71840: 65,
     65122: 510,
     62359: 4397,
     6897: 4458,
     145850: 1709,
     113285: 3161,
     195163: 15,
     28906: 2127,
     104290: 17585,
     80717: 871,
     14025: 3955,
     94979: 18312,
     167609: 36,
     156351: 50,
     87866: 4099,
     119027: 216,
     182562: 21,
     29250: 49,
     55417: 148,
     150811: 7,
     126677: 652,
     6411: 294,
     152270: 839,
     1624: 652,
     144225: 41,
     136571: 43,
     100416: 2188,
     142821: 668,
     143528: 508,
     191441: 1096,
     163474: 27430,
     21913: 73,
     131589: 61,
     30456: 513,
     17316: 83,
     17007: 9396,
     111761: 97,
     181782: 1821,
     93595: 73,
     2250: 1022,
     137384: 888,
     88510: 4074,
     50754: 1817,
     196996: 490,
     99774: 204,
     192295: 7243,
     60336: 778,
     170917: 246,
     61087: 319,
     178391: 59,
     126090: 165,
     191331: 28,
     82978: 262,
     166198: 67,
     51608: 747,
     152502: 6017,
     106268: 6,
     103659: 5386,
     73307: 29,
     117118: 2126,
     103490: 21,
     131356: 16,
     158482: 1053,
     67456: 2142,
     18073: 280,
     167295: 696,
     43958: 57,
     32433: 750,
     66354: 242,
     153953: 20255,
     36840: 265,
     2065: 11,
     137608: 16571,
     55733: 11,
     133952: 628,
     141210: 3580,
     86671: 3154,
     117505: 319,
     63622: 4015,
     180011: 102,
     135234: 7,
     34326: 7,
     89720: 8420,
     171905: 4006,
     133238: 3264,
     143311: 232,
     137911: 2423,
     103512: 412,
     40950: 651,
     142383: 208,
     89091: 428,
     69865: 9880,
     191747: 6466,
     91611: 2454,
     169852: 13454,
     167231: 2137,
     128571: 17571,
     36828: 2268,
     135648: 1269,
     14650: 4233,
     13044: 1780,
     115569: 171,
     67297: 13148,
     165742: 7722,
     46522: 579,
     32970: 78,
     15487: 144,
     37557: 185,
     9651: 51,
     103302: 7,
     192760: 70,
     162655: 734,
     23589: 1520,
     95380: 1332,
     15751: 3281,
     14366: 37224,
     190524: 2937,
     84132: 2311,
     28529: 7299,
     30165: 3104,
     20986: 1791,
     115292: 3338,
     147281: 4892,
     21648: 7449,
     163261: 1047,
     107774: 118,
     63179: 8568,
     132923: 84,
     11413: 589,
     131865: 4515,
     137113: 9445,
     24897: 6683,
     47468: 7366,
     167391: 1457,
     182816: 2029,
     169674: 9407,
     170061: 582,
     76791: 3963,
     145456: 7343,
     163511: 1723,
     82588: 97,
     41763: 10052,
     197313: 8075,
     186543: 43902,
     119467: 833,
     32375: 358,
     23925: 306,
     45881: 1746,
     116454: 3728,
     74491: 4466,
     114836: 4132,
     170679: 346,
     103099: 865,
     165570: 22636,
     152267: 667,
     189447: 23,
     82052: 2303,
     84103: 684,
     141531: 27209,
     66345: 1914,
     198617: 23765,
     54017: 1507,
     168506: 1285,
     172690: 12,
     111546: 2337,
     190509: 1867,
     1628: 40,
     66855: 186,
     8194: 990,
     49016: 24064,
     126491: 15088,
     99737: 5278,
     108447: 220,
     6924: 1019,
     79990: 78432,
     183434: 10889,
     72149: 2918,
     32234: 13059,
     43841: 2461,
     6227: 2411,
     87073: 10198,
     22905: 1073,
     105568: 3018,
     90883: 23324,
     26759: 26645,
     80839: 17400,
     39170: 801,
     101776: 9790,
     10076: 16249,
     126212: 2790,
     49889: 5830,
     125866: 15,
     144012: 1876,
     192700: 78,
     176665: 1936,
     174384: 271,
     40335: 2043,
     72170: 2056,
     26781: 176,
     49336: 1276,
     148311: 1480,
     5256: 2481,
     118094: 2636,
     47508: 1404,
     196948: 4685,
     151983: 156,
     59768: 1947,
     17665: 2414,
     78708: 19023,
     99250: 416,
     132926: 311,
     165607: 2817,
     35838: 42,
     87919: 3760,
     149577: 367,
     115802: 8507,
     184690: 150,
     91155: 45,
     137115: 1068,
     125226: 4024,
     44111: 15617,
     42090: 1425,
     136589: 2457,
     102663: 203,
     187573: 215,
     96858: 491,
     108407: 268,
     97151: 229,
     64646: 2859,
     36261: 13965,
     171125: 18201,
     13420: 733,
     7753: 9,
     133595: 1292,
     72404: 1391,
     9374: 900,
     30694: 2590,
     169502: 6599,
     155209: 625,
     63979: 361,
     91082: 948,
     122945: 3497,
     100: 57,
     43701: 337,
     130824: 355,
     69378: 3721,
     135858: 2036,
     44960: 644,
     128543: 5259,
     197365: 3402,
     196617: 48,
     65819: 6788,
     155992: 9,
     546: 4028,
     191110: 23,
     187347: 2459,
     160886: 86,
     98756: 1684,
     154504: 188,
     164690: 8976,
     15114: 2897,
     120758: 391,
     29518: 1394,
     6267: 3192,
     184738: 648,
     2000: 285,
     161164: 3022,
     158066: 9,
     67349: 3520,
     99857: 121,
     81965: 9529,
     172361: 249,
     140583: 2976,
     94422: 4723,
     18675: 6554,
     107615: 831,
     123757: 25,
     57437: 16,
     71966: 437,
     158499: 70,
     52945: 234,
     71678: 12,
     164137: 161,
     124622: 781,
     173425: 6887,
     123: 2623,
     172629: 5871,
     27213: 154,
     87918: 335,
     155012: 41,
     17422: 534,
     181700: 1182,
     4263: 80,
     180102: 674,
     53419: 2074,
     94921: 1597,
     55681: 25,
     49469: 9385,
     146122: 314,
     6355: 232,
     130204: 20,
     25038: 7,
     122711: 902,
     53562: 560,
     1129: 13,
     164276: 750,
     104570: 386,
     115524: 198,
     191115: 1857,
     116395: 10,
     157947: 3271,
     158642: 2128,
     188268: 299,
     5257: 449,
     168673: 30,
     99371: 753,
     98376: 812,
     13374: 15594,
     78762: 696,
     130028: 127,
     69948: 772,
     58179: 373,
     64451: 17,
     164110: 435,
     66169: 1632,
     68998: 143,
     190905: 3170,
     25947: 2436,
     39584: 523,
     65531: 1920,
     50652: 1287,
     3389: 549,
     33106: 93,
     58652: 10,
     112967: 263,
     164372: 363,
     186463: 913,
     101764: 621,
     191951: 30,
     35029: 15,
     111324: 63,
     94257: 28,
     103852: 1399,
     150282: 17,
     178911: 556,
     151969: 340,
     116198: 2532,
     151735: 9137,
     69529: 372,
     102738: 19310,
     176149: 514,
     51237: 68,
     68527: 123,
     159416: 12797,
     56935: 651,
     5850: 5017,
     5204: 4333,
     147613: 1426,
     781: 50,
     151764: 95,
     54236: 974,
     65346: 228,
     81230: 290,
     191001: 162,
     32268: 673,
     5537: 115,
     197469: 10,
     160800: 477,
     113851: 39,
     8996: 2321,
     184366: 527,
     167420: 225,
     133084: 1155,
     4235: 887,
     71597: 2884,
     175017: 823,
     55231: 1943,
     13143: 845,
     172241: 152,
     80221: 61,
     81580: 6050,
     120900: 10740,
     129081: 7691,
     65625: 29,
     11430: 38,
     37489: 24,
     153409: 16,
     112450: 6622,
     53564: 102,
     148435: 5766,
     66533: 298,
     197145: 15,
     44806: 384,
     20779: 7362,
     166616: 191,
     152363: 555,
     197236: 11,
     46636: 306,
     10325: 315,
     175808: 379,
     164475: 2357,
     49647: 6381,
     92672: 155,
     177121: 762,
     105024: 887,
     170525: 103,
     15115: 169,
     149954: 7085,
     146507: 13542,
     46784: 514,
     116420: 1792,
     167394: 405,
     101420: 326,
     191188: 384,
     163080: 5293,
     124494: 5086,
     61616: 2226,
     164932: 43,
     97891: 967,
     188857: 11137,
     193920: 5300,
     118940: 2779,
     51892: 64,
     118998: 7,
     89206: 1713,
     97764: 303,
     147312: 6838,
     161062: 46,
     102982: 227,
     82711: 436,
     45344: 9745,
     79989: 1018,
     146210: 323,
     23136: 473,
     49259: 2000,
     18312: 3153,
     123777: 21227,
     192543: 2377,
     112267: 348,
     161287: 13,
     154768: 357,
     166099: 780,
     183422: 710,
     170684: 50,
     33654: 613,
     182510: 732,
     129874: 243,
     159546: 1313,
     97521: 213,
     96667: 3289,
     144408: 760,
     182484: 1759,
     160496: 76,
     184372: 800,
     143734: 944,
     184482: 330,
     68569: 6,
     171576: 108,
     61042: 112,
     128845: 70,
     194749: 710,
     160251: 51,
     42341: 2663,
     40159: 305,
     43577: 303,
     114633: 48,
     159881: 4740,
     73016: 55,
     191173: 758,
     77464: 8605,
     83487: 25,
     73816: 24,
     70153: 560,
     95680: 18819,
     10271: 10,
     94836: 64,
     12108: 47,
     106712: 23,
     130220: 9454,
     160968: 511,
     161638: 653,
     50791: 130,
     80332: 1052,
     178215: 143,
     39587: 1545,
     55029: 4856,
     63919: 140,
     23167: 387,
     46452: 362,
     124913: 153,
     107129: 561,
     35830: 2917,
     116427: 109,
     185056: 35,
     39490: 1793,
     113369: 2155,
     79254: 178,
     18528: 180,
     87971: 647,
     40035: 3412,
     167020: 68,
     91547: 6,
     138586: 36,
     160362: 12,
     41407: 207,
     185117: 1081,
     61043: 324,
     35018: 103,
     103489: 357,
     182682: 3904,
     82044: 43,
     40820: 1740,
     82829: 77,
     157175: 17,
     160026: 18,
     141594: 20,
     113407: 1011,
     136983: 6,
     175822: 30,
     72987: 924,
     17675: 552,
     135566: 123,
     123500: 59,
     182040: 247,
     33721: 1421,
     190637: 354,
     117131: 142,
     155709: 239,
     118678: 32,
     132959: 13,
     122615: 1865,
     60853: 475,
     55405: 83,
     104894: 6,
     177155: 1256,
     14094: 1252,
     39395: 89,
     149999: 142,
     17832: 226,
     114458: 1240,
     104427: 244,
     45239: 2311,
     187706: 645,
     90642: 4329,
     196263: 990,
     37613: 2139,
     164173: 146,
     196072: 1485,
     59389: 68,
     89068: 1537,
     191506: 1715,
     55809: 860,
     92937: 3379,
     58654: 2287,
     162488: 381,
     84415: 118,
     128890: 134,
     61642: 110,
     16495: 481,
     17836: 139,
     139971: 510,
     29944: 234,
     70832: 19980,
     156594: 2463,
     31524: 5557,
     187180: 39,
     157772: 172,
     71179: 5338,
     128029: 641,
     40757: 26,
     28251: 325,
     169093: 434,
     94972: 133,
     103402: 1291,
     8618: 6741,
     14361: 3476,
     43155: 47,
     169196: 82,
     115804: 27,
     197653: 280,
     88722: 1782,
     51842: 1596,
     150802: 12,
     111246: 24,
     42617: 4237,
     74060: 7265,
     79623: 274,
     115015: 107,
     1159: 20,
     191973: 111,
     102760: 107,
     49750: 1215,
     39367: 5071,
     160765: 18,
     173166: 575,
     187535: 4453,
     56853: 789,
     98768: 314,
     56292: 12,
     168547: 169,
     171623: 2799,
     53280: 9,
     152558: 98,
     65213: 111,
     40760: 304,
     53965: 3158,
     111299: 1212,
     7290: 6056,
     113054: 199,
     101386: 756,
     67831: 91,
     48253: 32,
     7094: 4894,
     145925: 1041,
     143639: 2104,
     514: 27,
     161780: 1123,
     92695: 57,
     112876: 9184,
     31801: 96,
     102072: 1969,
     61720: 3545,
     97838: 1636,
     158389: 23,
     139706: 509,
     169697: 629,
     152414: 1008,
     137150: 328,
     101653: 128,
     22043: 270,
     29928: 2447,
     113822: 47,
     60343: 423,
     136989: 125,
     112239: 4379,
     135470: 417,
     78564: 235,
     103443: 4323,
     9974: 1024,
     63210: 2671,
     152312: 12570,
     84906: 503,
     197477: 2856,
     10653: 27666,
     153623: 379,
     7394: 1960,
     23298: 380,
     82136: 63,
     117230: 13294,
     175385: 2281,
     142304: 82,
     195827: 808,
     194726: 4903,
     15344: 11018,
     170172: 10166,
     181716: 7407,
     115388: 3508,
     31750: 1583,
     79859: 3224,
     119402: 4942,
     158914: 3048,
     117734: 971,
     72320: 4356,
     152736: 3112,
     28202: 533,
     176304: 554,
     67548: 1182,
     22086: 6875,
     172473: 294,
     28166: 43,
     167476: 2562,
     88567: 404,
     8118: 1653,
     45438: 246,
     105300: 2279,
     159451: 14236,
     152646: 3293,
     30317: 1309,
     65862: 1113,
     105874: 1424,
     17131: 11866,
     190645: 779,
     11007: 1734,
     72950: 40,
     42878: 1205,
     25144: 2641,
     89839: 8636,
     105241: 1359,
     166371: 294,
     151498: 88,
     171939: 606,
     24784: 1493,
     68724: 4377,
     22570: 1750,
     33482: 236,
     144567: 109,
     166851: 10698,
     98750: 275,
     11855: 9230,
     58643: 84,
     79701: 967,
     155833: 5617,
     103400: 1662,
     61714: 11248,
     44458: 184,
     117004: 2350,
     138015: 203,
     132047: 15,
     119538: 4887,
     39332: 699,
     156108: 12045,
     76360: 1789,
     116966: 3631,
     107044: 558,
     130100: 1073,
     41841: 19154,
     46501: 1332,
     53498: 6587,
     96665: 510,
     196172: 8374,
     142114: 1047,
     165892: 115,
     121875: 9448,
     62498: 3019,
     22986: 213,
     197866: 132,
     159459: 1298,
     51744: 2288,
     38769: 74,
     8030: 9886,
     30047: 171,
     163866: 855,
     70021: 1481,
     8848: 1240,
     80041: 7783,
     29786: 381,
     44570: 14451,
     26861: 2942,
     137350: 3098,
     64132: 28931,
     110606: 4522,
     67227: 43,
     78579: 2193,
     148865: 125,
     103358: 2600,
     41689: 3582,
     72905: 11347,
     111828: 343,
     141250: 575,
     156627: 175,
     93073: 1244,
     191627: 1887,
     43971: 18812,
     2746: 291,
     ...}



## calculate probability and conditional probability


```python
# cauculate probability of each word. calculate word_occurrence_pos{} and word_occurrence_neg{}
def get_probability(word_dict, word_dictionary, wordSet, word_occurrence, word_occurrence_pos, word_occurrence_neg):
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
def get_condition_m_estimate(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data):
    tmp = word_dictionary[word]
    conditional_probability_pos = 0
    conditional_probability_neg = 0
    if tmp in word_occurrence_pos.keys():
        conditional_probability_pos = float(word_occurrence_pos[tmp] + 1.5)/ float(size_of_data + 3)
    else: conditional_probability_pos = float(1.5)/ float(size_of_data + 3)
    if tmp in word_occurrence_neg.keys():
        conditional_probability_neg = float(word_occurrence_neg[tmp] + 1.5)/ float(size_of_data + 3)
    else: conditional_probability_neg = float(1.5)/ float(size_of_data + 3)
    return conditional_probability_pos, conditional_probability_neg
```


```python
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
                # m estimate
                conditional_probability_pos, conditional_probability_neg \
                = get_condition_m_estimate(word, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg, size_of_data)
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
    = get_probability(train_dict_final, word_dictionary, wordSet, word_occurrence,\
                  word_occurrence_pos, word_occurrence_neg)
    score.append(get_accuracy(test_dict_final, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg))
print(score)
mean_score = 0
for i in score:
     mean_score += i
print(float(mean_score)/ len(score))
```

    [0.9022, 0.901, 0.903, 0.905, 0.9238]
    0.907


### result of laplace smoothing
round 1:
[0.9, 0.8866, 0.8894, 0.888, 0.931]
0.899
round 2
[0.9, 0.8988, 0.8964, 0.8902, 0.9338]
0.90384

### result of m estimate smoothing
round 1
[0.905, 0.8964, 0.9014, 0.8958, 0.9268]
0.9050799999999999
round 2
[0.9022, 0.901, 0.903, 0.905, 0.9238]
0.907

m estimate smoothing have better result

## Derive Top 10 words that predicts positive and negative class


```python
def get_top_10(word_dict, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg):
    top_10_pos = PQueue()
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
    know
    end
    years
    ever
    man
    acting
    go
    makes
    back
    better
    
    
    Top 10 words that predicts negative
    something
    thing
    end
    character
    two
    many
    go
    im
    watching
    great


## Using the test dataset calculate the final accuracy.


```python
# final accuracy using m estimate smoothing
word_occurrence, word_occurrence_pos, word_occurrence_neg, doc_number \
    = get_probability(train_to_dict_tmp, word_dictionary, wordSet, word_occurrence,\
                  word_occurrence_pos, word_occurrence_neg)
   
score = get_accuracy(train_to_dict_tmp, word_dictionary, word_occurrence, word_occurrence_pos, word_occurrence_neg)
score

```




    0.91176



final accuracy using m estimate smoothing = 0.91176


```python

```
