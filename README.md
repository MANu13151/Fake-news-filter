# Fake-news-filter

I Have created a model that will take news from dataset and will tell if the news is fake or true. This will ensure no rumours will spread 

import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


data = pd.read_excel('fake.xlsx')

data['Label'] = data['Label'].str.strip().str.lower()

data['thought'] = data['Label'].apply(lambda x: 1 if x == 'correct' else 0)

data.to_excel('ufake.xlsx', index=False)

print(data.head())


swiss = WordNetLemmatizer()
nltk.download('stopwords')
stopwords = stopwords.words('English')
nltk.download('wordnet')
data1 = pd.read_excel('ufake.xlsx')
def clean_row(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]',' ',row)

    token = row.split()
    news = [swiss.lemmatize(word) for word in token if not word in stopwords]

    cleaniing = ' '.join(news)
    return(cleaniing)

data1['Headline'] = data1['Headline'].apply(lambda x: clean_row(x))


Vector = TfidfVectorizer(max_features=10000,lowercase=False,ngram_range=(1,3))
x = data1['Headline']
y = data1['thought']
print(y)

train_data, test_data , train_label , test_label = train_test_split(x,y, test_size = 0.2 , random_state = 0)

vec_train_data = Vector.fit_transform(train_data)
vec_train_data = vec_train_data.toarray()

vec_test_data = Vector.transform(test_data)
vec_test_data = vec_test_data.toarray()
vec_train_data.shape,vec_test_data.shape


train_data = pd.DataFrame(vec_train_data, columns = Vector.get_feature_names_out())
test_data = pd.DataFrame(vec_test_data, columns= Vector.get_feature_names_out())

cliff = MultinomialNB()
cliff.fit(train_data , train_label)
y_pred = cliff.predict(test_data)

score = accuracy_score(test_label , y_pred)
print( 'the accuracy is ' , score*100 , '%')

txt = ''
news = clean_row(txt)


vec_news = Vector.transform([news])
pred = cliff.predict(vec_news)
print(pred)

if(pred == 1):
	print("correct news !")
else:
	print("Fake news ! ")
