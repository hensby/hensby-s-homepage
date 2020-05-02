---
date: 2020-5-1
title: Review rating
---

# Term Project Data mining 
## Hengchao Wang 1001778272

{{% staticref "files/Game Review.ipynb" "newtab" %}}Download my files{{% /staticref %}}

[Link of my website](http://35.223.125.6:5000/)

### Reference. 
BoardGameGeek Reviews Baseline Model
https://www.kaggle.com/ellpeeaxe/boardgamegeek-reviews-baseline-model

Word2vec In Supervised NLP Tasks. Shortcut
https://www.kaggle.com/vladislavkisin/word2vec-in-supervised-nlp-tasks-shortcut/comments

Cuz the scale of the dataset is super big. Cannot use one hot expression to exprese words and sentences. It will cause the curse of dimensionality. Which means the matrix is big and sparse to be compute. So I decide to use Word2Vec word embedding model to reduce dimension of matrix. I have two references. The link is shown above. 

The based task of this question is a regression problem. The imput data is 300-dimensional word vector, output is the prediction of rate for each review.


```python
import numpy as np 
import pandas as pd 
import nltk
import re,string,unicodedata
import seaborn as sns
import gensim
import sklearn

from pandas import Series
from wordcloud import WordCloud,STOPWORDS
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from gensim.models import word2vec, Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import ensemble
```

## Get data from csv


```python
# get review and rating columns
review_path = 'bgg-13m-reviews.csv'

data = pd.read_csv(review_path, usecols=[2,3])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>Currently, this sits on my list as my favorite...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>I know it says how many plays, but many, many ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove null comment
def remove_nan(data):
    data['comment']=data['comment'].fillna('null')
    data = data[~data['comment'].isin(['null'])]
    data = data.reset_index(drop=True)
    return data
data = remove_nan(data)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rate</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>currently , thi sit list favorit game .</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
      <td>know say mani plays , many , mani uncounted. l...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>never tire thi game .. awesom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>thi probabl best game ever played. requir thin...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.0</td>
      <td>fantast game. got hook game .</td>
    </tr>
  </tbody>
</table>
</div>



**This is data describtion. The number of review is 2.637756e+06**


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.637756e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.852070e+00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.775769e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.401300e-45</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+01</td>
    </tr>
  </tbody>
</table>
</div>



## Data preprocessing

For data preprocessing I using **tokenizer()** from **NLTK** library to tokenize the words. Load stopword from **NLTK** and load html strips from **beautifulsoup4** library. Use regular expression to remove them and some special characters.


```python
#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
```


```python
#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
data['comment']=data['comment'].apply(remove_between_square_brackets)
```


```python
#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
data['comment']=data['comment'].apply(remove_special_characters)
```


```python
#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
data['comment']=data['comment'].apply(simple_stemmer)
```


```python
#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
data['comment']=data['comment'].apply(remove_stopwords)
```

    {"you'd", "that'll", 'other', 'any', "won't", "you're", 'have', 'yourselves', 'about', 'm', 'were', 'our', 'than', 'their', 'haven', 'being', 'over', 't', 'been', 'against', 'again', 'we', 'most', 'doesn', 'so', 'yourself', "aren't", 'mustn', 'under', 'just', 'down', 'ma', 'with', 'until', 'isn', 'don', 'shan', "shouldn't", 'myself', "you've", 'having', 'has', 'between', 'because', 'was', 'yours', 'nor', 'am', 'through', 'his', 'as', 'few', 'but', 'and', 'before', 'itself', 'hers', 'during', "mustn't", 'y', 'doing', 'an', "you'll", 'they', 'hasn', 'did', 'each', "couldn't", 'ours', 'weren', 'hadn', 'there', 'then', "doesn't", 'that', 'this', 'needn', 'no', 'i', 'aren', 'too', 'once', 'you', 'themselves', 'her', 'these', 'll', 'won', 'out', 'how', "it's", 'herself', 'to', 'when', 'o', 'my', 'of', 'into', "didn't", "hadn't", 'very', 'him', 'what', 'now', 'who', 'are', 'if', 'in', 'above', 'why', 'all', 'off', 'where', 'd', 'didn', 'couldn', 'while', 'does', 'she', 'wasn', 'theirs', 'the', 'your', "should've", 'by', 'up', 'whom', 'a', "weren't", 'same', "hasn't", 'mightn', "shan't", 'some', 'from', 'below', 're', 'which', 'those', "don't", "mightn't", 'will', 'its', 'only', "needn't", 'himself', 's', 'more', 'such', 'not', 'he', 'on', 'own', "she's", 'is', "haven't", 'be', 've', 'further', 'do', 'should', 'them', 'had', "wasn't", 'me', 'both', 'shouldn', 'or', 'can', 'for', 'ain', 'it', 'ourselves', "isn't", "wouldn't", 'here', 'at', 'after', 'wouldn'}


**After we remove the stopword we need to remove the empty review again because come short review after remove stopword will change into empty.**


```python
data = remove_nan(data)
data.to_csv('data_after_remove_st.csv', header=False, index=False, encoding = 'utf-8')
```


```python
columns = ['rate', 'comment']
data = pd.read_csv('data_after_remove_st.csv',names = columns)
```


```python
data['comment'] = data.comment.str.lower()
data['document_sentences'] = data.comment.str.split('.') 
# data['tokenized_sentences'] = data['document_sentences']
data['tokenized_sentences'] = list(map(lambda sentences:list(map(nltk.word_tokenize, sentences)),data.document_sentences))  
data['tokenized_sentences'] = list(map(lambda sentences: list(filter(lambda lst: lst, sentences)), data.tokenized_sentences))
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rate</th>
      <th>comment</th>
      <th>document_sentences</th>
      <th>tokenized_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>993001</th>
      <td>7.0</td>
      <td>good deduct game , go wrong question answer wr...</td>
      <td>[good deduct game , go wrong question answer w...</td>
      <td>[[good, deduct, game, ,, go, wrong, question, ...</td>
    </tr>
    <tr>
      <th>1965460</th>
      <td>6.0</td>
      <td>thi reason simpl area control game nice mechan...</td>
      <td>[thi reason simpl area control game nice mecha...</td>
      <td>[[thi, reason, simpl, area, control, game, nic...</td>
    </tr>
    <tr>
      <th>273330</th>
      <td>7.8</td>
      <td>awesom game. sleeved .</td>
      <td>[awesom game,  sleeved , ]</td>
      <td>[[awesom, game], [sleeved]]</td>
    </tr>
    <tr>
      <th>579587</th>
      <td>7.2</td>
      <td>thi everyth want bang ! , except much streamli...</td>
      <td>[thi everyth want bang ! , except much streaml...</td>
      <td>[[thi, everyth, want, bang, !, ,, except, much...</td>
    </tr>
    <tr>
      <th>740450</th>
      <td>6.0</td>
      <td>fun parti trivia game .</td>
      <td>[fun parti trivia game , ]</td>
      <td>[[fun, parti, trivia, game]]</td>
    </tr>
  </tbody>
</table>
</div>



### Challenge 1
**Here is a hint: Because the String[] cannot save as csv. The tokenized_sentences after save into csv will change the format into String and cannot load again.** This is one of a challenge I met. At the first few round of training Word2Vec model. The final accuracy is super low. I check the word expression of each word. The output from Word2Vec is less than 0.0001. That means that these word almost doesn't appear in the dataset. That doesn't make sence. So I check the model. The model.wv.vocab.keys() is small too and the vocabelory are latters, not words. So it must be the split problem or the format problem. So I check the type of each variable. The type of "tokenized_sentences" is changes. After google the issue. I found the point is you cannot save string[] in csv.

I wrote the wrong code as a comment in next 2 cells.


```python
# data.to_csv("data_after_pre.csv",sep=',',index=False, encoding = 'utf-8')
```


```python
# data = pd.read_csv('data_after_pre.csv')
```

The next cell will not be run when I train the Word2Vec. I train the Word2Vec model by using the whole Dataset.
The next cell will be run when I train the regression model. Cuz the computation I have only can use 50k reviews to train the regression model. So I use 10k and 50k reviews and compare them.


```python
# Take the top 10k after random ordering
data = data.reindex(np.random.permutation(data.index))[:100000]
```


```python
# split the data into training data and test data.
train, test, y_train, y_test = train_test_split(data, data['rate'], test_size=.2)
```


```python
type(train.tokenized_sentences[993001])
```




    list




```python
#Collecting a vocabulary
voc = []
for sentence in train.tokenized_sentences:
    voc.extend(sentence)
#     print(sentence)

print("Number of sentences: {}.".format(len(voc)))
print("Number of rows: {}.".format(len(train)))
```

    Number of sentences: 237600.
    Number of rows: 80000.



```python
voc[:10]
```




    [['(', 'vanilla', 'game', 'only'],
     ['play',
      'beyond',
      'black',
      ',',
      'mechan',
      'chang',
      'present',
      'expans',
      'might',




## Word2Vec model train, save and load

The number of feature in my Word2Vec model is 300. The matrix using one-hot expression is about 150k * 2.6M. Curse of dimensionality is gone.


```python
# word2vector
num_features = 300    
min_word_count = 3     # Frequency < 3 will not be count in.
num_workers = 16       
context = 8           
downsampling = 1e-3   

# Initialize and train the model
W2Vmodel = Word2Vec(sentences=voc, sg=1, hs=0, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                    sample=downsampling, negative=5, iter=6)
```


```python
model_voc = set(W2Vmodel.wv.vocab.keys()) 
print(len(model_voc))
```

    151488



```python
# model save
W2Vmodel.save("Word2Vec2")
```


```python
# model load
W2Vmodel = Word2Vec.load('Word2Vec2')
```

### Challenge 2

Train the model sentence by sentence is more accurate than the whole review. Cuz the length of the sentence are similar so that the feature of each input is similar. So I did not remove '.' when I remove noise character. That's come from comparison.


```python
def sentence_vectors(model, sentence):
    #Collecting all words in the text
#     print(sentence)
    sent_vector = np.zeros(model.vector_size, dtype="float32")
    if sentence == [[]] or sentence == []  :
        return sent_vector
    words=np.concatenate(sentence)
#     words = sentence
    #Collecting words that are known to the model
    model_voc = set(model.wv.vocab.keys()) 
#     print(len(model_voc))

    # Use a counter variable for number of words in a text
    nwords = 0
    # Sum up all words vectors that are know to the model
    for word in words:
        if word in model_voc: 
            sent_vector += model[word]
            nwords += 1.

    # Now get the average
    if nwords > 0:
        sent_vector /= nwords
    return sent_vector
```


```python
train['sentence_vectors'] = list(map(lambda sen_group:
                                      sentence_vectors(W2Vmodel, sen_group),
                                      train.tokenized_sentences))
test['sentence_vectors'] = list(map(lambda sen_group:
                                    sentence_vectors(W2Vmodel, sen_group), 
                                    test.tokenized_sentences))
```

    /home/sxy/anaconda3/envs/ML/lib/python3.6/site-packages/ipykernel_launcher.py:19: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).



```python
def vectors_to_feats(df, ndim):
    index=[]
    for i in range(ndim):
        df[f'w2v_{i}'] = df['sentence_vectors'].apply(lambda x: x[i])
        index.append(f'w2v_{i}')
    return df[index]
```


```python
X_train = vectors_to_feats(train, 300)
X_test = vectors_to_feats(test, 300)
```


```python
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
train.to_csv('train_w2v_100k.csv')
test.to_csv('test_w2v_100k.csv')
```


```python
train = pd.read_csv('train_w2v_1000k.csv').drop(columns = 'Unnamed: 0')
test = pd.read_csv('test_w2v_1000k.csv').drop(columns = 'Unnamed: 0')
X_train = train.drop(columns = 'rate')
X_test = test.drop(columns = 'rate')
y_train = train.rate
y_test = test.rate
```


```python
X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>w2v_0</th>
      <th>w2v_1</th>
      <th>w2v_2</th>
      <th>w2v_3</th>
      <th>w2v_4</th>
      <th>w2v_5</th>
      <th>w2v_6</th>
      <th>w2v_7</th>
      <th>w2v_8</th>
      <th>w2v_9</th>
      <th>...</th>
      <th>w2v_290</th>
      <th>w2v_291</th>
      <th>w2v_292</th>
      <th>w2v_293</th>
      <th>w2v_294</th>
      <th>w2v_295</th>
      <th>w2v_296</th>
      <th>w2v_297</th>
      <th>w2v_298</th>
      <th>w2v_299</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.064735</td>
      <td>-0.043941</td>
      <td>-0.391560</td>
      <td>0.194273</td>
      <td>0.038023</td>
      <td>-0.062682</td>
      <td>-0.003358</td>
      <td>-0.220116</td>
      <td>0.118839</td>
      <td>-0.210516</td>
      <td>...</td>
      <td>-0.113587</td>
      <td>0.034951</td>
      <td>-0.048320</td>
      <td>-0.084418</td>
      <td>-0.016730</td>
      <td>0.116862</td>
      <td>-0.006845</td>
      <td>0.039291</td>
      <td>0.216906</td>
      <td>-0.068584</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.108961</td>
      <td>-0.058336</td>
      <td>-0.318453</td>
      <td>0.191389</td>
      <td>0.005011</td>
      <td>-0.072080</td>
      <td>0.031846</td>
      <td>-0.165923</td>
      <td>0.149237</td>
      <td>-0.112924</td>
      <td>...</td>
      <td>-0.165357</td>
      <td>0.068865</td>
      <td>-0.048133</td>
      <td>-0.099376</td>
      <td>-0.037351</td>
      <td>0.075134</td>
      <td>0.002659</td>
      <td>0.027652</td>
      <td>0.179799</td>
      <td>-0.091966</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.037381</td>
      <td>-0.105713</td>
      <td>-0.223736</td>
      <td>0.230134</td>
      <td>-0.058320</td>
      <td>0.015869</td>
      <td>0.157899</td>
      <td>-0.395270</td>
      <td>0.309151</td>
      <td>-0.230134</td>
      <td>...</td>
      <td>-0.326233</td>
      <td>-0.136316</td>
      <td>-0.017143</td>
      <td>-0.049190</td>
      <td>0.112281</td>
      <td>0.129845</td>
      <td>-0.085892</td>
      <td>-0.036840</td>
      <td>0.082894</td>
      <td>-0.135891</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.073528</td>
      <td>-0.035113</td>
      <td>-0.295632</td>
      <td>0.210620</td>
      <td>0.030401</td>
      <td>0.020846</td>
      <td>0.056935</td>
      <td>-0.036678</td>
      <td>0.192137</td>
      <td>-0.148260</td>
      <td>...</td>
      <td>-0.098527</td>
      <td>-0.010514</td>
      <td>-0.077920</td>
      <td>-0.030762</td>
      <td>0.024684</td>
      <td>0.083572</td>
      <td>0.047993</td>
      <td>0.049659</td>
      <td>0.240063</td>
      <td>-0.042669</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19995</th>
      <td>-0.046645</td>
      <td>-0.102675</td>
      <td>-0.378800</td>
      <td>0.147131</td>
      <td>-0.031686</td>
      <td>-0.006655</td>
      <td>0.074540</td>
      <td>-0.169375</td>
      <td>0.167047</td>
      <td>-0.096239</td>
      <td>...</td>
      <td>-0.178984</td>
      <td>-0.024323</td>
      <td>-0.100185</td>
      <td>-0.013699</td>
      <td>-0.016214</td>
      <td>0.134338</td>
      <td>0.052931</td>
      <td>0.011761</td>
      <td>0.187225</td>
      <td>-0.053723</td>
    </tr>
    <tr>
      <th>19996</th>
      <td>-0.041224</td>
      <td>-0.043504</td>
      <td>-0.181815</td>
      <td>0.241858</td>
      <td>-0.087189</td>
      <td>0.012512</td>
      <td>0.010387</td>
      <td>-0.268597</td>
      <td>0.120241</td>
      <td>-0.173561</td>
      <td>...</td>
      <td>0.044585</td>
      <td>0.042733</td>
      <td>0.204142</td>
      <td>-0.184483</td>
      <td>-0.097213</td>
      <td>0.072322</td>
      <td>-0.009312</td>
      <td>0.044582</td>
      <td>0.361448</td>
      <td>-0.079877</td>
    </tr>
    <tr>
      <th>19997</th>
      <td>0.016242</td>
      <td>-0.070195</td>
      <td>-0.235203</td>
      <td>0.257024</td>
      <td>0.072520</td>
      <td>-0.119281</td>
      <td>-0.028535</td>
      <td>-0.243668</td>
      <td>0.219881</td>
      <td>-0.223677</td>
      <td>...</td>
      <td>-0.117014</td>
      <td>0.083126</td>
      <td>0.004575</td>
      <td>-0.047602</td>
      <td>0.008902</td>
      <td>0.131965</td>
      <td>-0.026648</td>
      <td>-0.042032</td>
      <td>0.170854</td>
      <td>-0.087977</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>-0.088338</td>
      <td>0.009536</td>
      <td>-0.190505</td>
      <td>0.197417</td>
      <td>-0.081475</td>
      <td>-0.028796</td>
      <td>0.044730</td>
      <td>-0.118943</td>
      <td>0.050266</td>
      <td>-0.045812</td>
      <td>...</td>
      <td>-0.132129</td>
      <td>0.220461</td>
      <td>0.029903</td>
      <td>-0.025690</td>
      <td>0.050592</td>
      <td>-0.100897</td>
      <td>0.093619</td>
      <td>0.050197</td>
      <td>0.166418</td>
      <td>-0.089344</td>
    </tr>
    <tr>
      <th>19999</th>
      <td>-0.036372</td>
      <td>-0.024210</td>
      <td>-0.283933</td>
      <td>0.139767</td>
      <td>0.035674</td>
      <td>-0.090993</td>
      <td>0.046099</td>
      <td>-0.137280</td>
      <td>0.160993</td>
      <td>-0.107587</td>
      <td>...</td>
      <td>-0.109823</td>
      <td>0.013688</td>
      <td>-0.014184</td>
      <td>-0.152064</td>
      <td>-0.037780</td>
      <td>0.003016</td>
      <td>0.037712</td>
      <td>-0.028141</td>
      <td>0.235616</td>
      <td>-0.055234</td>
    </tr>
  </tbody>
</table>
<p>20000 rows × 300 columns</p>
</div>



# implement different regression model
I implement 4 regression model and compare them with Root Mean Square Error (RMSE) and Mean absolute error(MAE).

**RMSE:** Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. It can tells you how concentrated the data is around the line of best fit. 

**MAE:**  Mean absolute error (MAE) is a measure of errors between paired observations expressing the same phenomenon. It is thus an arithmetic average of the absolute errors |ei|=|yi-xi|, where yi is the prediction and xi the true value. 

### Linear regression model
Linear regression is a basic and commonly used type of predictive analysis. Parameter calculation of linear equation using least squares method.

[Linear regression introduction](https://machinelearningmastery.com/linear-regression-for-machine-learning/)


```python
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
lr_y_predict=model_lr.predict(X_test)
y_test = np.array(y_test)
```


```python
# (RMSE)
rmse = np.sqrt(mean_squared_error(y_test,lr_y_predict))

# (MAE)
mae = mean_absolute_error(y_test, lr_y_predict)

print('linear_regression_rmse = ', rmse)
print('linear_regression_mae = ', mae)
```

    linear_regression_rmse =  1.5985795417920345
    linear_regression_mae =  1.2179818164120766



```python
joblib.dump(model_lr, 'save/model_lr.pkl')

# model_lr = joblib.load('save/model_lr_1000k.pkl')
```




    ['save/model_lr.pkl']



### SVR model
Support vector regression(SVR) is an application of support vector machine(SVM) to regression problem.

Regression is like looking for the internal relationship of a bunch of data. Regardless of whether the pile of data consists of several categories, a formula is obtained to fit these data. When a new coordinate value is given, a new value can be obtained. So for SVR, it is to find a face or a function, and you can fit all the data (that is, all data points, regardless of the type, the closest distance from the data point to the face or function)

[SVR introduction introduction](https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2)


```python
model_svm = SVR()
model_svm.fit(X_train, y_train)
```




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)




```python
svm_y_predict=model_svm.predict(X_test)
```


```python
# (RMSE)
rmse = np.sqrt(mean_squared_error(y_test,svm_y_predict))

# (MAE)
mae = mean_absolute_error(y_test, svm_y_predict)

print('svm_rmse = ', rmse)
print('svm_mae = ', mae)
```

    svm_rmse =  1.4967321667740556
    svm_mae =  1.1245787830283758



```python
joblib.dump(model_lr, 'save/model_svm.pkl')

```




    ['save/model_svm.pkl']



### Bayesian Ridge model
In the Bayesian viewpoint, we formulate linear regression using probability distributions rather than point estimates. The response, y, is not estimated as a single value, but is assumed to be drawn from a probability distribution.

The output, y is generated from a normal (Gaussian) Distribution characterized by a mean and variance. The mean for linear regression is the transpose of the weight matrix multiplied by the predictor matrix. The variance is the square of the standard deviation σ (multiplied by the Identity matrix because this is a multi-dimensional formulation of the model).

[Bayesian Ridge introduction](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7)


```python
model_bayes_ridge = BayesianRidge()
model_bayes_ridge.fit(X_train, y_train)
```




    BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None,
                  compute_score=False, copy_X=True, fit_intercept=True,
                  lambda_1=1e-06, lambda_2=1e-06, lambda_init=None, n_iter=300,
                  normalize=False, tol=0.001, verbose=False)




```python
bayes_y_predict = model_bayes_ridge.predict(X_test)
```


```python
# (RMSE)
rmse = np.sqrt(mean_squared_error(y_test,bayes_y_predict))

# (MAE)
mae = mean_absolute_error(y_test, bayes_y_predict)

print('BayesianRidge_rmse = ', rmse)
print('BayesianRidge_mae = ', mae)
```

    BayesianRidge_rmse =  1.5980023290295695
    BayesianRidge_mae =  1.2175385536747287



```python
joblib.dump(model_bayes_ridge, 'save/model_bayes.pkl')

```




    ['save/model_bayes.pkl']



### Random Forest Regression model

Random forest is a bagging technique and not a boosting technique. The trees in random forests are run in parallel. There is no interaction between these trees while building the trees.

The throught of Random Forest Regression is using the Boosting and ensemble in decision tree. In the lecture mentioned.

[Random Forest Regression introduction](https://towardsdatascience.com/random-forest-and-its-implementation-71824ced454f)


```python
model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=20)
model_random_forest_regressor.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=20, n_jobs=None, oob_score=False,
                          random_state=None, verbose=0, warm_start=False)




```python
random_forest_y_predict = model_random_forest_regressor.predict(X_test)
```


```python
# (RMSE)
rmse = np.sqrt(mean_squared_error(y_test,random_forest_y_predict))

# (MAE)
mae = mean_absolute_error(y_test, random_forest_y_predict)

print('BayesianRidge_rmse = ', rmse)
print('BayesianRidge_mae = ', mae)
```

    BayesianRidge_rmse =  1.6054778376150676
    BayesianRidge_mae =  1.2233214206573564



```python
joblib.dump(model_random_forest_regressor, 'save/model_random_forest.pkl')

```




    ['save/model_random_forest.pkl']



### Predict function for one review with four model


```python
def predict(text):
    model_lr = joblib.load('save/model_lr.pkl')
    model_svm = joblib.load('save/model_svm.pkl')
    model_random_forest_regressor = joblib.load('save/model_random_forest.pkl')
    model_bayes_ridge = joblib.load('save/model_bayes.pkl')
    data = {'comment': Series(text)}
    data = pd.DataFrame(data)
    print(data)
    data['comment'] = data['comment'].apply(remove_between_square_brackets)
    data['comment'] = data['comment'].apply(remove_special_characters)
    data['comment'] = data['comment'].apply(simple_stemmer)
    data['comment'] = data['comment'].apply(remove_stopwords)

    data['comment'] = data.comment.str.lower()
    data['document_sentences'] = data.comment.str.split('.')
    data['tokenized_sentences'] = data['document_sentences']
    data['tokenized_sentences'] = list(
        map(lambda sentences: list(map(nltk.word_tokenize, sentences)), data.document_sentences))
    data['tokenized_sentences'] = list(
        map(lambda sentences: list(filter(lambda lst: lst, sentences)), data.tokenized_sentences))
    print(data)
    # sentence = data['tokenized_sentences'][0]
    W2Vmodel = Word2Vec.load("Word2Vec2")

    data['sentence_vectors'] = list(map(lambda sen_group:
                                        sentence_vectors(W2Vmodel, sen_group),
                                        data.tokenized_sentences))
    text = vectors_to_feats(data, 300)
    print(text)
    lr_y_predict = model_lr.predict(text)
    svm_y_predict = model_svm.predict(text)
    bayes_y_predict = model_bayes_ridge.predict(text)
    random_forest_y_predict = model_random_forest_regressor.predict(text)

    return lr_y_predict, svm_y_predict, random_forest_y_predict, bayes_y_predict

```


```python
print(predict(["This is a great game.  I've even got a number of non game players enjoying it.  Fast to learn and always changing.",
        "This is a great game.  I've even got a number of non game players enjoying it.  Fast to learn and always changing."]))
```

                                                 comment
    0  This is a great game.  I've even got a number ...
    1  This is a great game.  I've even got a number ...
                                                 comment  \
    0  thi great game ive even got number non game pl...   
    1  thi great game ive even got number non game pl...   
    
                                      document_sentences  \
    0  [thi great game ive even got number non game p...   
    1  [thi great game ive even got number non game p...   
    
                                     tokenized_sentences  
    0  [[thi, great, game, ive, even, got, number, no...  
    1  [[thi, great, game, ive, even, got, number, no...  


    /home/sxy/anaconda3/envs/ML/lib/python3.6/site-packages/ipykernel_launcher.py:19: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).


          w2v_0     w2v_1     w2v_2     w2v_3     w2v_4     w2v_5     w2v_6  \
    0 -0.052897 -0.077122 -0.441616  0.210372  0.019172 -0.060663  0.048674   
    1 -0.052897 -0.077122 -0.441616  0.210372  0.019172 -0.060663  0.048674   
    
          w2v_7     w2v_8     w2v_9  ...   w2v_290  w2v_291  w2v_292   w2v_293  \
    0 -0.169603  0.132948 -0.137659  ... -0.135482   0.0026 -0.05121 -0.148072   
    1 -0.169603  0.132948 -0.137659  ... -0.135482   0.0026 -0.05121 -0.148072   
    
        w2v_294  w2v_295   w2v_296   w2v_297   w2v_298  w2v_299  
    0 -0.029361  0.08649 -0.070255 -0.040144  0.108867 -0.01677  
    1 -0.029361  0.08649 -0.070255 -0.040144  0.108867 -0.01677  
    
    [2 rows x 300 columns]
    (array([8.09318704, 8.09318704]), array([8.09318704, 8.09318704]), array([8.41475, 8.41475]), array([8.06230953, 8.06230953]))

