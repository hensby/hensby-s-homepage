---
date: 2020-4-14
title: Assignment3
---
{{% staticref "files/Hengchao_03.ipynb" "newtab" %}}Download my files{{% /staticref %}}


# Assignment 3 Naive bayes classifier
### Hengchao Wang 1001778272
 
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
Some of my thoughts of this part come from my own homework in my Machine learning class. 


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
word_dictionary
```




    {'acquarium': 0,
     'panicked': 1,
     'scammers': 2,
     'orchidshowgirls': 3,
     'dumbbrainachingly': 4,
     'ludlam': 5,
     'jiuan': 6,
     'xb1': 7,
     'mahoganygirl': 8,
     'manturnedflying': 9,
     'julietthey': 10,
     'mob': 11,
     'leybr': 12,
     'cheatingstupid': 13,
     'gulfax': 14,
     'unholiness': 15,
     'elli': 16,
     'suzannes': 17,
     'chingling': 18,
     'swears': 19,
     'slip': 20,
     'moatschariots': 21,
     'piglias': 22,
     'eaternwestern': 23,
     'cannibalising': 24,
     'zorica': 25,
     'randell': 26,
     'vibration': 27,
     'frank': 28,
     'cheapozoid': 29,
     'emotionallyladen': 30,
     '1there': 31,
     'oergon': 32,
     'homewrecker': 33,
     'duelspace': 34,
     'mythical': 35,
     'digimon': 36,
     'generated': 37,
     'yadda': 38,
     'wm': 39,
     'samoilovitch': 40,
     'aglae': 41,
     'hodgsons': 42,
     'covey': 43,
     'suzu': 44,
     'manifestations': 45,
     'searchlight': 46,
     'sono': 47,
     'scenefor': 48,
     '10dirstuart': 49,
     'homossexual': 50,
     'casomai': 51,
     'streatchy': 52,
     'brett': 53,
     '18931951': 54,
     'voyeurfriendly': 55,
     'audiencegoers': 56,
     'sceptered': 57,
     'beginging': 58,
     'intergender': 59,
     'allentype': 60,
     'snowballing': 61,
     'fearinducing': 62,
     'listed': 63,
     'mindblowing': 64,
     'pakeeza': 65,
     'cupoftea': 66,
     'jeonghwa': 67,
     'drones': 68,
     'doggoned': 69,
     'rochahaving': 70,
     'mered': 71,
     'shoresituated': 72,
     'stuns': 73,
     'pasty': 74,
     'maximise': 75,
     'yearsbr': 76,
     'scmaltzy': 77,
     'policemenwoman': 78,
     'screwloose': 79,
     'disappointedif': 80,
     'nivola': 81,
     'skipfeature': 82,
     'highlyannoying': 83,
     'entwines': 84,
     'vacillates': 85,
     '136': 86,
     'warningshot': 87,
     'caughtup': 88,
     'surveyor': 89,
     'arent': 90,
     'redevelopment': 91,
     'homolka': 92,
     'carnewhile': 93,
     'genoas': 94,
     'layfield': 95,
     'plotsummary': 96,
     'wetdreamturned': 97,
     'factwith': 98,
     'heinlein': 99,
     'complianceas': 100,
     'decorated': 101,
     'technicians': 102,
     'passer': 103,
     'inferenced': 104,
     'selfsufficiency': 105,
     'harmful': 106,
     'cameradirector': 107,
     'pseudooriginal': 108,
     'pompas': 109,
     'chaplins': 110,
     'torturing': 111,
     'retrieves': 112,
     'accidentand': 113,
     'haudenasaunee': 114,
     'subjection': 115,
     'safedeposit': 116,
     'vw': 117,
     '3040mph': 118,
     'exploredone': 119,
     'dreamsno': 120,
     'influencethe': 121,
     'easternbloc': 122,
     'pasts': 123,
     'castwith': 124,
     'mockshakespearean': 125,
     'maternally': 126,
     'pietistic': 127,
     'chennai': 128,
     'comngm0312feature2index': 129,
     'balloons': 130,
     'eeeww': 131,
     'pillladen': 132,
     'barmitzvahs': 133,
     'vance': 134,
     'childmolesters': 135,
     'maudehe': 136,
     'grifter': 137,
     'overeager': 138,
     'dietrichsons': 139,
     'danceduet': 140,
     'jagi': 141,
     'shrouding': 142,
     'eighthgraders': 143,
     'hennessey': 144,
     'tubbss': 145,
     'canyons': 146,
     'containment': 147,
     'entry1992': 148,
     'schnappmann': 149,
     'myerss': 150,
     'juergen': 151,
     'janis': 152,
     'stored': 153,
     'fulcicut': 154,
     'churchburning': 155,
     'pertfectly': 156,
     'spotlight': 157,
     'polyamory': 158,
     'goodindicates': 159,
     'mariette': 160,
     'poster': 161,
     'demeaning': 162,
     'thankyouled': 163,
     'churchbr': 164,
     'eightyday': 165,
     'waissbluth': 166,
     'puppo': 167,
     'wellwith': 168,
     'tuileries': 169,
     'pankcaking': 170,
     'stepmom': 171,
     'jaisa': 172,
     'popstarplayed': 173,
     'tackleberry': 174,
     'victorianism': 175,
     'whoredup': 176,
     'incestridden': 177,
     'fierstein': 178,
     'rephrasing': 179,
     'delia': 180,
     'rapp': 181,
     'krauts': 182,
     'overlookedbut': 183,
     'jags': 184,
     'filmmuseum': 185,
     'blistering': 186,
     'futomaki': 187,
     'revolutionized': 188,
     'actionwhen': 189,
     'nickelndime': 190,
     'ironcurtain': 191,
     'chokingly': 192,
     'berate': 193,
     'occur': 194,
     'dissociate': 195,
     'spontaneousappearing': 196,
     'ub': 197,
     'defadubbing': 198,
     'linkmake': 199,
     'anotherthe': 200,
     'nosedives': 201,
     'ceremonially': 202,
     'sneller': 203,
     'troubles': 204,
     'herothen': 205,
     'vacuous': 206,
     'actorscombine': 207,
     'spiritand': 208,
     'beatlemania': 209,
     'asylumis': 210,
     'sensibilitya': 211,
     'smarttalk': 212,
     'totter': 213,
     'oceanides': 214,
     'colums': 215,
     'bladder': 216,
     'salaciousor': 217,
     'chiavi': 218,
     'homeward': 219,
     'tennen': 220,
     'vouched': 221,
     'bothas': 222,
     'inexcusable': 223,
     'redshorted': 224,
     'pennick': 225,
     'aips': 226,
     'dashnak': 227,
     'frankels': 228,
     'writermurder': 229,
     'gouched': 230,
     'xplanation': 231,
     'mostbelievable': 232,
     'pedals': 233,
     'volkos': 234,
     'cobane': 235,
     'smacks': 236,
     'alf': 237,
     'worksbr': 238,
     'cumpsty': 239,
     'kenans': 240,
     'trecking': 241,
     'predjudice': 242,
     'chubby': 243,
     'refining': 244,
     'ultraprincipled': 245,
     'pseudoad': 246,
     'jeweled': 247,
     '3day': 248,
     'dufosse': 249,
     'viewersbr': 250,
     'wiggleroom': 251,
     'griswolds': 252,
     'scheissencritter': 253,
     'sunnywarmquiet': 254,
     'iriquois': 255,
     'halloweentown': 256,
     'anarchistic': 257,
     'ikeda': 258,
     'raintree': 259,
     'greenerys': 260,
     'withheld': 261,
     'fillin': 262,
     'concluded': 263,
     'motorfashion': 264,
     'intensively': 265,
     'mixedup': 266,
     'hallway': 267,
     'dominate': 268,
     'velociraptor': 269,
     'werecrocodile': 270,
     'ummmmm': 271,
     'clubbed': 272,
     'walnuts': 273,
     'outdrawn': 274,
     'superskinny': 275,
     'feddy': 276,
     'mellifluous': 277,
     'subpredator': 278,
     'barbarian': 279,
     'woelfel': 280,
     'haseltine': 281,
     'imyself': 282,
     'hardtoswallow': 283,
     'umeckichan': 284,
     'creation': 285,
     'leadings': 286,
     'dustfinger': 287,
     'jesserobert': 288,
     'dvda': 289,
     'hubschmid': 290,
     'shita': 291,
     'cinematelevision': 292,
     'stimulant': 293,
     'mstified': 294,
     'wellmash': 295,
     'rockn': 296,
     'weirdfaced': 297,
     '3movie': 298,
     'decouteau': 299,
     'duhhhhhhhhhhhhh': 300,
     'onbut': 301,
     'electrocution': 302,
     'unpriveledged': 303,
     'battlemake': 304,
     'format': 305,
     'punchabledoofy': 306,
     'xwife': 307,
     'jerkiest': 308,
     'kittenand': 309,
     'sinus': 310,
     'steadycams': 311,
     'tobr': 312,
     'annistons': 313,
     'saoirise': 314,
     'marga': 315,
     'neighbourhoods': 316,
     'writerdirectory': 317,
     '3film': 318,
     'tropes': 319,
     'gildenlow': 320,
     'multiculturalism': 321,
     'ultramini': 322,
     'wsnt': 323,
     'yawwwwwnnnnnn': 324,
     'chump': 325,
     'westernizationis': 326,
     'selby': 327,
     'developingthere': 328,
     'endfields': 329,
     'recompose': 330,
     'customized': 331,
     'blackburns': 332,
     'flagwavers': 333,
     'raffinjerry': 334,
     'surroundsus': 335,
     '232002': 336,
     'bullheaded': 337,
     'reasontom': 338,
     'chano': 339,
     'everintense': 340,
     'wajdas': 341,
     'linsday': 342,
     '910br': 343,
     'untraceable': 344,
     'kerntype': 345,
     'universo': 346,
     'comedicfantastic': 347,
     'strasberg': 348,
     'fleck': 349,
     'spidershaped': 350,
     'rycart': 351,
     'hahahaaa': 352,
     'personand': 353,
     'comedydramahorror': 354,
     'patricks': 355,
     'dragonlslayer': 356,
     'eskil': 357,
     'jindabynes': 358,
     'shouts': 359,
     'blazing': 360,
     'afters': 361,
     'essaysp': 362,
     'aplus': 363,
     'loymake': 364,
     'szubanski': 365,
     'chihara': 366,
     'blacklisting': 367,
     'hackamns': 368,
     'discovers': 369,
     'scooterriding': 370,
     'invergordon': 371,
     'haywards': 372,
     'catastrophealthough': 373,
     'cinecittas': 374,
     'engulfed': 375,
     'leick': 376,
     'lateappearing': 377,
     'lightsensitive': 378,
     'sleptwalked': 379,
     'sholem': 380,
     'ltrange': 381,
     'crawling': 382,
     'daimond': 383,
     'welldelivered': 384,
     'euroreptarland': 385,
     'hiruta': 386,
     'killernow': 387,
     'kaneshiro': 388,
     'passivenatural': 389,
     'jarring': 390,
     'trashindustrial': 391,
     'tossers': 392,
     'columbos': 393,
     'mostra': 394,
     'powerslave': 395,
     'heisse': 396,
     'yuzos': 397,
     'allred': 398,
     'wobbly': 399,
     'amezin': 400,
     'pets': 401,
     'mankato': 402,
     'randkin': 403,
     'politically': 404,
     'everest': 405,
     'ohmagods': 406,
     'gamblinghouse': 407,
     'transplanted': 408,
     'nonmusicians': 409,
     'sisk': 410,
     'albeniz': 411,
     'whoopdedoodles': 412,
     'bergere': 413,
     'neartheknuckle': 414,
     'brest': 415,
     'examined': 416,
     'williamson': 417,
     'overserious': 418,
     'idolto': 419,
     'glint': 420,
     'tribulation': 421,
     'consuming': 422,
     'ferdos': 423,
     'killernotdeadyet': 424,
     'posest': 425,
     'petrescu': 426,
     'fitzgerald': 427,
     'austarlia': 428,
     'supercompetent': 429,
     'landis': 430,
     'tell': 431,
     'smashedtobits': 432,
     'defraud': 433,
     'resort': 434,
     'sordid': 435,
     'bitchslaps': 436,
     'bina': 437,
     'pygmallion': 438,
     'nonmenacing': 439,
     'vintagetype': 440,
     'familybut': 441,
     'temmink': 442,
     'imaginationchallenged': 443,
     'lapsesespecially': 444,
     'badguy': 445,
     'swallowbr': 446,
     'planatsatan': 447,
     'outdoor': 448,
     'woof': 449,
     'empd': 450,
     'onk': 451,
     'liberals': 452,
     'terribleyet': 453,
     'berkoff': 454,
     'flipping': 455,
     'actingplenty': 456,
     'braveness': 457,
     'sexdance': 458,
     'minigrotesques': 459,
     'khmer': 460,
     'italowestern': 461,
     'entertainmentbut': 462,
     'priyankas': 463,
     'flocks': 464,
     'consciousnessaltering': 465,
     'vladek': 466,
     'pusridden': 467,
     'manequineclass': 468,
     'affairit': 469,
     'thomash': 470,
     'kidwell': 471,
     'gether': 472,
     'burgialien': 473,
     'rapeif': 474,
     'wolgfang': 475,
     'enjoyablebr': 476,
     'backwaaards': 477,
     'sodomy': 478,
     'clitterhouse': 479,
     'colourscape': 480,
     'typewriters': 481,
     'playdohbeing': 482,
     'gorak': 483,
     'maletofemale': 484,
     'mukaddar': 485,
     'tgmb': 486,
     'stuffiness': 487,
     'promenant': 488,
     'exampleon': 489,
     'choiceby': 490,
     'nip': 491,
     'kozs': 492,
     'nabokovian': 493,
     'bogmeister': 494,
     'loompas': 495,
     'effeil': 496,
     'toolkit': 497,
     'monobr': 498,
     'protagonistvillain': 499,
     'browbeat': 500,
     'underweigh': 501,
     'los': 502,
     'christoper': 503,
     'blackton': 504,
     'spadeknife': 505,
     'stonewashed': 506,
     'sludgelike': 507,
     'teenish': 508,
     'annoyingkid': 509,
     'straighttotelevision': 510,
     'torturefake': 511,
     'creepiness': 512,
     'nirohoffman': 513,
     'mcmillian': 514,
     'filmshelms': 515,
     'gothicretro': 516,
     'mabe': 517,
     '96yearold': 518,
     'exeterjohn': 519,
     'igrayne': 520,
     'westernism': 521,
     'bahadir': 522,
     'gotchs': 523,
     'machominded': 524,
     'photograpy': 525,
     'brawled': 526,
     'schoolapproved': 527,
     'arabbased': 528,
     'streaks': 529,
     'blogs': 530,
     'proddies': 531,
     'shultz': 532,
     'whites': 533,
     'curryloving': 534,
     'rooshus': 535,
     'unqiue': 536,
     'audiowise': 537,
     'entertainmentmovie': 538,
     'actressis': 539,
     'beerbellies': 540,
     'inimitable': 541,
     'informative': 542,
     '74th': 543,
     'boosted': 544,
     'combs': 545,
     'bisaya': 546,
     'hunebelle': 547,
     'breakdowncrying': 548,
     'chuckerman': 549,
     'historicbr': 550,
     'macrabe': 551,
     'girlonly': 552,
     'sutdying': 553,
     'affordablepopular': 554,
     'minots': 555,
     'harryhausenstyle': 556,
     'benneys': 557,
     'abra': 558,
     'bi1': 559,
     'coupard': 560,
     'dekha': 561,
     'teeheehee': 562,
     'brevity': 563,
     'kristanna': 564,
     'astral': 565,
     'perfomrnace': 566,
     'cockpit': 567,
     'scenesthe': 568,
     'midchase': 569,
     'f85': 570,
     'moonchild': 571,
     'undesired': 572,
     'superboxmart': 573,
     'leeli': 574,
     'flagwavin': 575,
     'draculas': 576,
     '4week': 577,
     'fertilizer': 578,
     'symbolisms': 579,
     'attentionseeking': 580,
     '123': 581,
     'timings': 582,
     'desertor': 583,
     '2rights': 584,
     'lopezbentley': 585,
     'gothically': 586,
     'dunnaway': 587,
     'morsels': 588,
     'standardstyle': 589,
     'formulaintofilm': 590,
     'riel': 591,
     'unacknowledged': 592,
     'alreadyreviewed': 593,
     'gorgeouslycrafted': 594,
     'compassing': 595,
     'work': 596,
     'collet': 597,
     'unnamed': 598,
     'grard': 599,
     'loveicantdowithoutu': 600,
     'czechoslovakian': 601,
     'alucard': 602,
     'salum': 603,
     'crashlanding': 604,
     'storyarch': 605,
     'figgy': 606,
     'justbr': 607,
     'convincingbr': 608,
     'sororities': 609,
     'scoundtrack': 610,
     'leseon': 611,
     'indianna': 612,
     'klutziness': 613,
     'zowie': 614,
     'whomwhat': 615,
     'vrai': 616,
     'dope': 617,
     'oshram3': 618,
     'maddie': 619,
     'boyrescuesanimal': 620,
     'volt': 621,
     'marioncrawford': 622,
     'wayive': 623,
     'shouldawouldacoulda': 624,
     '135m': 625,
     'ychosen': 626,
     'britisher': 627,
     'differentiation': 628,
     'toomeys': 629,
     'bannnister': 630,
     'satiresubsketch': 631,
     'vaszarys': 632,
     'supersecret': 633,
     'fantasticly': 634,
     'shame': 635,
     'coguard': 636,
     'hardhat': 637,
     'superstud': 638,
     'hayness': 639,
     'nupur': 640,
     'letzter': 641,
     'specialite': 642,
     'negitive': 643,
     'fincherpitt': 644,
     'flourishlisten': 645,
     'pixel': 646,
     'sixtyeight': 647,
     'cryor': 648,
     'dvdsubtitle': 649,
     'phillion': 650,
     'shortlived': 651,
     'vamgoo': 652,
     'sneers': 653,
     'tito': 654,
     'marsigleses': 655,
     'everthing': 656,
     'siragusa': 657,
     'spellings': 658,
     'acidrock': 659,
     'kdc': 660,
     'relaying': 661,
     'derrida': 662,
     'decentunfortunately': 663,
     'option': 664,
     'greedily': 665,
     'ecoplace': 666,
     'bugrade': 667,
     'countryengland': 668,
     'antidepression': 669,
     'sufferable': 670,
     '0000110': 671,
     'debenning': 672,
     'solicit': 673,
     'bernardo': 674,
     'audiencedone': 675,
     'coreographed': 676,
     'saddly': 677,
     'performanceis': 678,
     'blackout': 679,
     'alertan': 680,
     'convulse': 681,
     'goldwater': 682,
     'worst': 683,
     'ghandi': 684,
     'wriggling': 685,
     'balaban': 686,
     '40sgood': 687,
     'flicki': 688,
     'cargoload': 689,
     'selfimportant': 690,
     'rossa': 691,
     'carfilm': 692,
     'lollar': 693,
     'infrom': 694,
     'gamecube': 695,
     'excepton': 696,
     'metamoprhis': 697,
     'sneedeker': 698,
     'prig': 699,
     'rumanian': 700,
     'royals': 701,
     'bertulucci': 702,
     'dailes': 703,
     'depictedthe': 704,
     'sfsite': 705,
     'glaad': 706,
     'meeler': 707,
     'bristowbr': 708,
     'costumers': 709,
     'glimcher': 710,
     'activitylina': 711,
     '19421945': 712,
     'demonkicking': 713,
     'gestures': 714,
     'eckardt': 715,
     'sajid': 716,
     'hollywoood': 717,
     'matchings': 718,
     'flubbed': 719,
     'rusticcastle': 720,
     'espada': 721,
     'totality': 722,
     'ilayarajas': 723,
     'hillerman': 724,
     'actingsome': 725,
     '700800': 726,
     'padayappa': 727,
     'ghetoization': 728,
     'mothersince': 729,
     'discouraged': 730,
     'noapologies': 731,
     'lackofquality': 732,
     'bushleague': 733,
     'seonhwa': 734,
     'filmshes': 735,
     'bashfulness': 736,
     'naghib': 737,
     'womanbarbara': 738,
     'deceitful': 739,
     'interconnections': 740,
     'chessloving': 741,
     'mekull': 742,
     'boltaction': 743,
     'karriena': 744,
     'tractorone': 745,
     'scotchfree': 746,
     'spiderweb': 747,
     'fixes': 748,
     'covington': 749,
     'earthand': 750,
     'abortionscandal': 751,
     'barneying': 752,
     'humorist': 753,
     'yuletidejaded': 754,
     'shipiro': 755,
     'careerbest': 756,
     'jackal': 757,
     'crashand': 758,
     'taxi3': 759,
     'spacial': 760,
     'bigamy': 761,
     'overreaction': 762,
     'journalist': 763,
     'dancerswho': 764,
     'vegetas': 765,
     'baryshnikovs': 766,
     'lute': 767,
     'sasaki': 768,
     'leering': 769,
     'yeahhhhhhhhhhh': 770,
     'damnation': 771,
     'legitimated': 772,
     'jocksbr': 773,
     'mathieuconchita': 774,
     'leann': 775,
     'glamourous': 776,
     'barfing': 777,
     'propman': 778,
     'christmanish': 779,
     'mapother': 780,
     'readhead': 781,
     'panpeter': 782,
     'predating': 783,
     'awakeable': 784,
     'napster': 785,
     'easymoney': 786,
     'dangers': 787,
     'drools': 788,
     'sks': 789,
     'mid1934': 790,
     'othelo': 791,
     'ukjoking_apart_s1': 792,
     'worldlink': 793,
     'settinggood': 794,
     'vama': 795,
     'firesign': 796,
     'inebriated': 797,
     'bachelorwedding': 798,
     'rozemas': 799,
     'complying': 800,
     'theremin': 801,
     'koirala': 802,
     'bendixs': 803,
     'dacron': 804,
     'sukhvinder': 805,
     'preposterousness': 806,
     'alloriginal': 807,
     'astonish': 808,
     'chillsfull': 809,
     'tap': 810,
     'collected': 811,
     'agnieszkas': 812,
     'awfulwith': 813,
     'chlup': 814,
     'gallons': 815,
     'courtyard': 816,
     'fanatics': 817,
     'essanay': 818,
     'sandro': 819,
     'hartnell': 820,
     'acquited': 821,
     'monsterrevenge': 822,
     'gaffigan': 823,
     'girl77': 824,
     '40shighstyle': 825,
     'sugdens': 826,
     'ferfectly': 827,
     'bartholomewand': 828,
     'flivvers': 829,
     'bullfighters': 830,
     'bungholebr': 831,
     'congestion': 832,
     'outflanking': 833,
     '98minutes': 834,
     'virtuoso': 835,
     'baxterbirney': 836,
     'unpan': 837,
     'flees': 838,
     'zombiethat': 839,
     'hankerin': 840,
     'birchstyle': 841,
     '617': 842,
     'lb3': 843,
     'sitcoms': 844,
     'selfurination': 845,
     'maniaccam': 846,
     'fancast': 847,
     'juran': 848,
     'welby': 849,
     'circumscribed': 850,
     'rounder': 851,
     'babelfish': 852,
     'actiondirecting': 853,
     'veohmuch': 854,
     'spiderthings': 855,
     'nearnaked': 856,
     'covetous': 857,
     'deist': 858,
     'consolidate': 859,
     'grapeas': 860,
     'saif': 861,
     'plagiarise': 862,
     'alfonso': 863,
     'paybut': 864,
     'bader': 865,
     'coms': 866,
     'dorffs': 867,
     'mclaglan': 868,
     'testosteronedriven': 869,
     'stickler': 870,
     'childbirth': 871,
     'marlyn': 872,
     'disturbingwithout': 873,
     'pushing': 874,
     'filmbuff': 875,
     'galaxyquest': 876,
     'cultfollowings': 877,
     'rohner': 878,
     'becoems': 879,
     'beetleborgs': 880,
     'wagnerian': 881,
     'rancocas': 882,
     'calligraphic': 883,
     'overbroad': 884,
     'conservativeminded': 885,
     'loa': 886,
     'moviesid': 887,
     'latexfaced': 888,
     'amwe': 889,
     'generates': 890,
     'fuzzily': 891,
     'perimeter': 892,
     'negre': 893,
     'mote': 894,
     'autitlesfeaturesbuddies': 895,
     'trisexual': 896,
     'designatmosphere': 897,
     'fullycostumed': 898,
     'flee': 899,
     'honestthat': 900,
     'swaggarty': 901,
     'expressway': 902,
     'tvdvd': 903,
     'furthermorethe': 904,
     'nonbreakable': 905,
     'mindmeld': 906,
     'tavares': 907,
     'pharmasists': 908,
     'lieutenantthe': 909,
     'hatsu': 910,
     'stanch': 911,
     'pandeybachchanpandey': 912,
     'resit': 913,
     'peschi': 914,
     'heartsick': 915,
     'godgiven': 916,
     'malamaal': 917,
     'twocents': 918,
     'coprophagia': 919,
     'airplay': 920,
     'francophones': 921,
     'koyaanisqatsi': 922,
     'scads': 923,
     'aeriss': 924,
     'chadas': 925,
     'murvyn': 926,
     'rejectssequel': 927,
     'arrrgghhh': 928,
     'warheroandgreatmasterandcommander': 929,
     'edgeofyourseats': 930,
     'bocellis': 931,
     'pourpres': 932,
     'horable': 933,
     'onefiftieth': 934,
     'poonam': 935,
     'scholes': 936,
     'incantation': 937,
     'guiltily': 938,
     'vapors': 939,
     'hartwin': 940,
     'rogowskis': 941,
     'impressed': 942,
     'correlating': 943,
     'themremarkable': 944,
     'supportersits': 945,
     'giggle': 946,
     'insinuations': 947,
     'wellfare': 948,
     'realisticwitty': 949,
     'moviesweet': 950,
     'impart': 951,
     'meekerreduced': 952,
     'fallswith': 953,
     'selfderision': 954,
     'inflaming': 955,
     'filledout': 956,
     'indiains': 957,
     'albas': 958,
     'reexperienced': 959,
     'noirhard': 960,
     'labrador': 961,
     'motions': 962,
     'filmindustri': 963,
     'schygulla': 964,
     'ister': 965,
     'interconnecting': 966,
     'polito': 967,
     'butttwitcher': 968,
     'sistine': 969,
     'clarify': 970,
     'burchill': 971,
     'revoew': 972,
     'sevier': 973,
     'chinese': 974,
     'odysey': 975,
     'postlimelight': 976,
     'cerebrate': 977,
     'moonbeast': 978,
     'worldas': 979,
     'touchup': 980,
     'housesitter': 981,
     'ashwood': 982,
     'rsums': 983,
     'sjostroms': 984,
     'commitsbr': 985,
     'sugarplams': 986,
     'scripting': 987,
     'alltoooften': 988,
     'restitution': 989,
     'socialatorprostitute': 990,
     'macchio': 991,
     'preformances': 992,
     'chilean': 993,
     '1rating': 994,
     'andreja': 995,
     'semispiritual': 996,
     'experiencepretty': 997,
     'wwii': 998,
     'zenlike': 999,
     ...}




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




```python
word_count
```




    {64021: 128732,
     65945: 9451,
     148586: 37603,
     107161: 1423,
     91568: 7870,
     45439: 12170,
     136163: 1976,
     9575: 268,
     63: 348,
     131163: 1014,
     149210: 2181,
     17299: 1933,
     69457: 17,
     9001: 12752,
     148103: 12974,
     133032: 143,
     156130: 31,
     43380: 16174,
     35021: 31,
     198344: 6,
     138387: 20145,
     122767: 1076,
     105654: 721,
     132861: 20549,
     169326: 881,
     68168: 19,
     47980: 139,
     33753: 851,
     186336: 3252,
     25700: 123,
     13352: 1457,
     106627: 10270,
     149373: 1506,
     78690: 2158,
     166496: 1140,
     32732: 625,
     16952: 42,
     67944: 93,
     31519: 6420,
     10634: 155,
     94602: 6,
     108776: 1372,
     160802: 10879,
     24141: 2879,
     160947: 140,
     34000: 669,
     60819: 11,
     65221: 876,
     121005: 40,
     104003: 1791,
     108615: 423,
     7349: 47,
     83498: 28,
     180392: 28,
     146382: 3483,
     57345: 15762,
     6612: 54,
     10590: 29453,
     176326: 796,
     155799: 1264,
     87306: 8936,
     62684: 4544,
     12210: 95,
     166039: 1249,
     52892: 7,
     192419: 249,
     106442: 1540,
     24609: 1746,
     71504: 242,
     119565: 216,
     76391: 23,
     114311: 114,
     175334: 740,
     110088: 640,
     98093: 4483,
     7286: 855,
     60143: 11175,
     3053: 34858,
     56860: 5248,
     151717: 141,
     62602: 281,
     109808: 3132,
     82956: 86,
     164406: 355,
     199738: 89,
     131687: 133,
     87456: 92,
     172828: 5274,
     72489: 172,
     111027: 606,
     17854: 311,
     46674: 28,
     96660: 125,
     125789: 97,
     94414: 61,
     155092: 27239,
     58910: 7945,
     59928: 42,
     121091: 1459,
     173995: 2298,
     23176: 92,
     178859: 1795,
     35546: 707,
     115024: 4277,
     130515: 42,
     190801: 256,
     129555: 7,
     166652: 93,
     53779: 38,
     196605: 363,
     191609: 4841,
     18972: 168,
     1618: 2276,
     153670: 11569,
     89077: 3427,
     138749: 629,
     119988: 26,
     23178: 38,
     10305: 99,
     56803: 7091,
     193629: 500,
     137222: 1488,
     89828: 1507,
     68246: 101,
     152419: 115922,
     112090: 243,
     188866: 72,
     29888: 4794,
     70140: 3577,
     129351: 11930,
     18522: 13430,
     129383: 36274,
     36105: 23654,
     128526: 14921,
     135812: 23771,
     152642: 34118,
     41266: 13,
     5781: 4778,
     84897: 4024,
     57544: 17,
     162706: 611,
     114411: 275,
     5080: 160,
     55674: 118,
     114114: 22,
     99824: 2836,
     29313: 3311,
     69844: 2048,
     177209: 2932,
     180857: 2277,
     127233: 144,
     138783: 2619,
     19082: 39,
     153623: 5586,
     5581: 4292,
     173839: 59415,
     162742: 6649,
     82658: 360,
     181667: 5395,
     104391: 6,
     34124: 18,
     73116: 2506,
     34105: 829,
     169243: 248,
     80676: 6512,
     31664: 840,
     57806: 99,
     18001: 100,
     23250: 3760,
     153827: 531,
     156164: 34699,
     174600: 2240,
     165617: 12861,
     156433: 3698,
     17289: 3272,
     86314: 186,
     164730: 52,
     80653: 1188,
     158613: 23,
     48625: 1589,
     37074: 14,
     15261: 7585,
     33489: 5528,
     104567: 19392,
     75193: 2424,
     194874: 7485,
     123171: 114,
     14554: 910,
     165802: 50,
     176770: 3093,
     123541: 743,
     101671: 1212,
     157318: 36,
     84189: 198,
     7291: 6,
     123398: 4709,
     76621: 11,
     92722: 82,
     87667: 34,
     175734: 5950,
     73656: 1013,
     93454: 29,
     10896: 37,
     116346: 3728,
     136457: 5839,
     197150: 75,
     24974: 1368,
     103379: 44,
     154491: 176,
     151732: 1390,
     23349: 578,
     194124: 1090,
     170786: 32,
     181075: 4187,
     150490: 598,
     51925: 62,
     49001: 420,
     133261: 163,
     177706: 8,
     102488: 740,
     121621: 6872,
     148527: 3004,
     24737: 188,
     171649: 190,
     108108: 36299,
     98456: 523,
     176817: 107,
     128899: 8,
     70252: 1295,
     56569: 126,
     117382: 68,
     183568: 6443,
     19809: 85,
     176587: 1877,
     183981: 34,
     173357: 14,
     41865: 280,
     55649: 27,
     18774: 67,
     180501: 76,
     167242: 170,
     136952: 578,
     17144: 294,
     136935: 4714,
     58200: 109,
     192503: 65,
     78498: 510,
     159885: 4397,
     150304: 4458,
     172570: 1709,
     170750: 3161,
     8692: 15,
     117250: 2127,
     175170: 17585,
     142857: 871,
     159096: 3955,
     107762: 18312,
     167504: 36,
     56552: 50,
     68752: 4099,
     128269: 216,
     76428: 21,
     36322: 49,
     193517: 148,
     119385: 7,
     144504: 652,
     88876: 294,
     37951: 839,
     74500: 652,
     178015: 41,
     108350: 43,
     97263: 2188,
     28672: 668,
     90289: 508,
     163263: 1096,
     163681: 27430,
     144028: 73,
     23999: 61,
     72409: 513,
     76056: 83,
     189622: 9396,
     171052: 97,
     129086: 1821,
     113810: 73,
     123252: 1022,
     133710: 888,
     180656: 4074,
     123622: 1817,
     164947: 490,
     186646: 204,
     24457: 7243,
     170544: 778,
     183677: 246,
     175166: 319,
     130679: 59,
     195713: 165,
     187759: 28,
     13788: 262,
     179847: 67,
     137902: 747,
     89666: 6017,
     121858: 6,
     89729: 5386,
     69836: 29,
     16995: 2126,
     52080: 21,
     178025: 16,
     48576: 1053,
     84550: 2142,
     119294: 280,
     124434: 696,
     29756: 57,
     19496: 750,
     36890: 242,
     94660: 20255,
     110198: 265,
     180055: 11,
     149949: 16571,
     154348: 11,
     110585: 628,
     67953: 3580,
     21709: 3154,
     153078: 319,
     172819: 4015,
     6547: 102,
     113657: 7,
     67766: 7,
     45888: 8420,
     75803: 4006,
     24518: 3264,
     20086: 232,
     150494: 2423,
     177988: 412,
     49061: 651,
     9633: 208,
     29759: 428,
     125713: 9880,
     86776: 6466,
     145264: 2454,
     61388: 13454,
     181140: 2137,
     75889: 17571,
     104086: 2268,
     63148: 1269,
     126290: 4233,
     84593: 1780,
     117499: 171,
     13753: 13148,
     42985: 7722,
     171412: 579,
     140676: 78,
     182455: 144,
     145936: 185,
     111928: 51,
     105882: 7,
     140236: 70,
     10537: 734,
     151075: 1520,
     62361: 1332,
     40435: 3281,
     118490: 37224,
     3214: 2937,
     116104: 2311,
     65170: 7299,
     199688: 3104,
     36151: 1791,
     99876: 3338,
     139117: 4892,
     15143: 7449,
     122627: 1047,
     157747: 118,
     152491: 8568,
     20389: 84,
     178964: 589,
     107275: 4515,
     48799: 9445,
     63955: 6683,
     117402: 7366,
     91583: 1457,
     110959: 2029,
     190768: 9407,
     12567: 582,
     103647: 3963,
     76522: 7343,
     135226: 1723,
     49006: 97,
     5894: 10052,
     33299: 8075,
     65077: 43902,
     143566: 833,
     138191: 358,
     133131: 306,
     107183: 1746,
     109338: 3728,
     106303: 4466,
     41245: 4132,
     20383: 346,
     93928: 865,
     30426: 22636,
     108985: 667,
     91494: 23,
     102659: 2303,
     12066: 684,
     14404: 27209,
     180951: 1914,
     134116: 23765,
     171423: 1507,
     27834: 1285,
     137023: 12,
     118644: 2337,
     127135: 1867,
     109832: 40,
     90005: 186,
     110919: 990,
     183635: 24064,
     83588: 15088,
     40551: 5278,
     102331: 220,
     34501: 1019,
     87764: 78432,
     38428: 10889,
     90914: 2918,
     51016: 13059,
     53656: 2461,
     191731: 2411,
     114462: 10198,
     133890: 1073,
     113944: 3018,
     64083: 23324,
     135097: 26645,
     73340: 17400,
     37287: 801,
     119487: 9790,
     90492: 16249,
     164407: 2790,
     11582: 5830,
     179735: 15,
     65016: 1876,
     75934: 78,
     73519: 1936,
     100983: 271,
     19000: 2043,
     48870: 2056,
     55638: 176,
     78754: 1276,
     28014: 1480,
     142969: 2481,
     56730: 2636,
     48703: 1404,
     103652: 4685,
     171337: 156,
     35470: 1947,
     165770: 2414,
     127801: 19023,
     38482: 416,
     114261: 311,
     61360: 2817,
     17713: 42,
     40836: 3760,
     35774: 367,
     92898: 8507,
     121665: 150,
     180762: 45,
     38489: 1068,
     56848: 4024,
     17659: 15617,
     43500: 1425,
     66545: 2457,
     14763: 203,
     168162: 215,
     177163: 491,
     117624: 268,
     178314: 229,
     156473: 2859,
     133675: 13965,
     174240: 18201,
     164256: 733,
     179870: 9,
     114064: 1292,
     143009: 1391,
     80404: 900,
     83624: 2590,
     100349: 6599,
     84551: 625,
     80982: 361,
     40188: 948,
     189006: 3497,
     2989: 57,
     109501: 337,
     24048: 355,
     136333: 3721,
     48991: 2036,
     32452: 644,
     10721: 5259,
     63918: 3402,
     114818: 48,
     116834: 6788,
     29273: 9,
     1647: 4028,
     194339: 23,
     28472: 2459,
     175667: 86,
     11164: 1684,
     93824: 188,
     8785: 8976,
     138893: 2897,
     197545: 391,
     13167: 1394,
     101013: 3192,
     92059: 648,
     46404: 285,
     80636: 3022,
     43268: 9,
     61276: 3520,
     139468: 121,
     157757: 9529,
     40954: 249,
     33858: 2976,
     187928: 4723,
     29406: 6554,
     75101: 831,
     12411: 25,
     35978: 16,
     26944: 437,
     117779: 70,
     33885: 234,
     94934: 12,
     70070: 161,
     117412: 781,
     182679: 6887,
     68394: 2623,
     141458: 5871,
     115354: 154,
     175302: 335,
     191456: 41,
     111547: 534,
     94498: 1182,
     18142: 80,
     79649: 674,
     8475: 2074,
     155587: 1597,
     118595: 25,
     34536: 9385,
     97559: 314,
     43966: 232,
     79748: 20,
     90748: 7,
     89903: 902,
     144785: 560,
     142318: 13,
     1386: 750,
     62249: 386,
     51563: 198,
     194494: 1857,
     158706: 10,
     14789: 3271,
     37001: 2128,
     174199: 299,
     91062: 449,
     149833: 30,
     46246: 753,
     24282: 812,
     140820: 15594,
     197388: 696,
     113380: 127,
     38786: 772,
     91047: 373,
     183947: 17,
     170518: 435,
     9499: 1632,
     96081: 143,
     68685: 3170,
     118242: 2436,
     161094: 523,
     23920: 1920,
     97639: 1287,
     55937: 549,
     166636: 93,
     195587: 10,
     63031: 263,
     5723: 363,
     96762: 913,
     156636: 621,
     74572: 30,
     84202: 15,
     140735: 63,
     32543: 28,
     126520: 1399,
     173013: 17,
     126976: 556,
     78757: 340,
     116394: 2532,
     47000: 9137,
     185872: 372,
     143939: 19310,
     27473: 514,
     90064: 68,
     77700: 123,
     174420: 12797,
     124137: 651,
     431: 5017,
     62756: 4333,
     31511: 1426,
     165160: 50,
     143582: 95,
     58106: 974,
     30039: 228,
     141902: 290,
     80386: 162,
     141507: 673,
     22365: 115,
     160031: 10,
     41281: 477,
     62375: 39,
     175867: 2321,
     88576: 527,
     49732: 225,
     91296: 1155,
     81413: 887,
     133299: 2884,
     161057: 823,
     79780: 1943,
     75477: 845,
     29068: 152,
     169229: 61,
     4923: 6050,
     195440: 10740,
     60161: 7691,
     45791: 29,
     190814: 38,
     193188: 24,
     28387: 16,
     88110: 6622,
     169415: 102,
     67943: 5766,
     45476: 298,
     171142: 15,
     62260: 384,
     128293: 7362,
     36117: 191,
     118203: 555,
     106481: 11,
     170297: 306,
     138732: 315,
     126497: 379,
     92795: 2357,
     50202: 6381,
     21117: 155,
     71590: 762,
     91987: 887,
     165038: 103,
     69605: 169,
     182579: 7085,
     113842: 13542,
     157893: 514,
     32096: 1792,
     57230: 405,
     93289: 326,
     197569: 384,
     73749: 5293,
     146449: 5086,
     90423: 2226,
     127144: 43,
     20819: 967,
     104813: 11137,
     164784: 5300,
     41366: 2779,
     1198: 64,
     149616: 7,
     10186: 1713,
     55428: 303,
     62434: 6838,
     183367: 46,
     86201: 227,
     68415: 436,
     134321: 9745,
     61118: 1018,
     193015: 323,
     157575: 473,
     154800: 2000,
     150920: 3153,
     43578: 21227,
     119714: 2377,
     82757: 348,
     66967: 13,
     14017: 357,
     5056: 780,
     2708: 710,
     185438: 50,
     111194: 613,
     158022: 732,
     32112: 243,
     104168: 1313,
     188763: 213,
     88654: 3289,
     146842: 760,
     43504: 1759,
     21395: 76,
     85867: 800,
     83823: 944,
     123066: 330,
     86556: 6,
     52278: 108,
     30709: 112,
     38957: 70,
     32109: 710,
     157969: 51,
     171704: 2663,
     104789: 305,
     33380: 303,
     168880: 48,
     10596: 4740,
     114092: 55,
     9823: 758,
     112401: 8605,
     66765: 25,
     87862: 24,
     130996: 560,
     100312: 18819,
     101133: 10,
     147437: 64,
     19347: 47,
     88158: 23,
     195360: 9454,
     152961: 511,
     79624: 653,
     191458: 130,
     87450: 1052,
     138811: 143,
     192254: 1545,
     40681: 4856,
     145268: 140,
     119644: 387,
     99581: 362,
     182717: 153,
     73979: 561,
     183753: 2917,
     118116: 109,
     14287: 35,
     33400: 1793,
     53232: 2155,
     54843: 178,
     127791: 180,
     8815: 647,
     132140: 3412,
     158253: 68,
     155449: 6,
     120895: 36,
     123696: 12,
     7301: 207,
     188698: 1081,
     170859: 324,
     11720: 103,
     182877: 357,
     28221: 3904,
     112071: 43,
     44171: 1740,
     139638: 77,
     94348: 17,
     92850: 18,
     196165: 20,
     5806: 1011,
     181199: 6,
     56703: 30,
     169365: 924,
     39635: 552,
     71903: 123,
     67737: 59,
     171743: 247,
     181873: 1421,
     139120: 354,
     167125: 142,
     84216: 239,
     48780: 32,
     77557: 13,
     68368: 1865,
     85898: 475,
     97486: 83,
     84532: 6,
     179645: 1256,
     18859: 1252,
     188078: 89,
     176205: 142,
     83356: 226,
     187273: 1240,
     41333: 244,
     69184: 2311,
     62457: 645,
     65307: 4329,
     61403: 990,
     107765: 2139,
     125547: 146,
     127711: 1485,
     197666: 68,
     149018: 1537,
     132589: 1715,
     84460: 860,
     101297: 3379,
     114502: 2287,
     50008: 381,
     158912: 118,
     30228: 134,
     57168: 110,
     156071: 481,
     124153: 139,
     69877: 510,
     79525: 234,
     164036: 19980,
     60322: 2463,
     93288: 5557,
     21436: 39,
     193456: 172,
     1015: 5338,
     135405: 641,
     6665: 26,
     52100: 325,
     30300: 434,
     72191: 133,
     64038: 1291,
     22957: 6741,
     109918: 3476,
     41742: 47,
     56638: 82,
     6098: 27,
     138241: 280,
     177038: 1782,
     109375: 1596,
     161022: 12,
     182918: 24,
     39517: 4237,
     127887: 7265,
     80931: 274,
     19071: 107,
     166308: 20,
     120912: 111,
     39612: 107,
     18618: 1215,
     2545: 5071,
     111268: 18,
     20340: 575,
     173741: 4453,
     137846: 789,
     90680: 314,
     83033: 12,
     136633: 169,
     158795: 2799,
     180767: 9,
     35375: 98,
     171041: 111,
     29222: 304,
     64808: 3158,
     10782: 1212,
     33012: 6056,
     189496: 199,
     43698: 756,
     54921: 91,
     126080: 32,
     141537: 4894,
     158781: 1041,
     159625: 2104,
     118967: 27,
     168414: 1123,
     79957: 57,
     47399: 9184,
     158985: 96,
     89578: 1969,
     13749: 3545,
     51480: 1636,
     147594: 23,
     28494: 509,
     17546: 629,
     133450: 1008,
     193815: 328,
     36376: 128,
     68135: 270,
     16787: 2447,
     67880: 47,
     80731: 423,
     118347: 125,
     78487: 4379,
     96078: 417,
     79291: 235,
     86421: 4323,
     57593: 1024,
     9883: 2671,
     186860: 12570,
     11197: 503,
     88261: 2856,
     139091: 27666,
     181225: 379,
     163702: 1960,
     164072: 380,
     39732: 63,
     126479: 13294,
     191867: 2281,
     158590: 82,
     166539: 808,
     198844: 4903,
     48565: 11018,
     114595: 10166,
     88882: 7407,
     87879: 3508,
     88192: 1583,
     50363: 3224,
     182678: 4942,
     67666: 3048,
     103973: 971,
     94836: 4356,
     163928: 3112,
     149139: 533,
     44725: 554,
     124396: 1182,
     174387: 6875,
     184364: 294,
     128926: 43,
     81054: 2562,
     163096: 404,
     129254: 1653,
     29702: 246,
     179156: 2279,
     4757: 14236,
     140055: 3293,
     21063: 1309,
     106245: 1113,
     68865: 1424,
     37550: 11866,
     187707: 779,
     147200: 1734,
     39771: 40,
     166375: 1205,
     189927: 2641,
     106284: 8636,
     74740: 1359,
     142729: 294,
     21907: 88,
     48430: 606,
     51991: 1493,
     97878: 4377,
     116614: 1750,
     74879: 236,
     177720: 109,
     184578: 10698,
     194257: 275,
     143636: 9230,
     5245: 84,
     32798: 967,
     138357: 5617,
     7759: 1662,
     66588: 11248,
     8254: 184,
     104425: 2350,
     103192: 203,
     31277: 15,
     49581: 4887,
     24184: 699,
     171444: 12045,
     96408: 1789,
     108689: 3631,
     15909: 558,
     166628: 1073,
     42483: 19154,
     127411: 1332,
     64998: 6587,
     86970: 510,
     172149: 8374,
     184934: 1047,
     113859: 115,
     115891: 9448,
     110436: 3019,
     119879: 213,
     28787: 132,
     186651: 1298,
     124490: 2288,
     14841: 74,
     24829: 9886,
     33655: 171,
     36055: 855,
     179045: 1481,
     135794: 1240,
     141136: 7783,
     88607: 381,
     101808: 14451,
     19842: 2942,
     92121: 3098,
     55072: 28931,
     9192: 4522,
     190437: 43,
     122527: 2193,
     69239: 125,
     163072: 2600,
     87202: 3582,
     109336: 11347,
     7439: 343,
     177493: 575,
     44288: 175,
     107210: 1244,
     9287: 1887,
     108889: 18812,
     166258: 291,
     50864: 753,
     ...}



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
