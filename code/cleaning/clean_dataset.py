import pandas as pd
from kkltk.kin_kir_stopwords import stopwords   # check https://github.com/Andrews2017/kkltk for more detailed information about how to use kkltk package

stopset_kin = stopwords.words('kinyarwanda') 

# loading the data
data = pd.read_csv('../data/KINNEWS/raw/train.csv')

# Cleaning the data (preprocessing)
# Removing the special characters and urls
data.title = data.title.str.replace('[^A-Za-z\s\’\-]+', '')
data.content = data.content.str.replace('[^A-Za-z\s\’\-]+', '')
data.title = data.title.str.replace('[\n]+', '')
data.content = data.content.str.replace('[\n]+', '')
data.title = data.title.str.replace('^https?:\/\/.*[\r\n]*', '')
data.content = data.content.str.replace('^https?:\/\/.*[\r\n]*', '')

# Removing the stopwords
data['title'] = data['title'].apply(lambda x: ' '.join([item.lower() for item in str(x).split() if item not in stopset_kin]))
data['content'] = data['content'].apply(lambda x: ' '.join([item.lower() for item in str(x).split() if item not in stopset_kin]))

# Save the cleaned dataset
data.to_csv("../data/KINNEWS/cleaned/train.csv", index=False)
