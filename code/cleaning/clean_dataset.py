import pandas as pd

stopset_kin = {'aba', 'abo', 'aha', 'aho', 'ari', 'ati', 'aya', 'ayo', 'ba', 'baba', 'babo', 'bari', 'be', 'bo', 'bose',
           'bw', 'bwa', 'bwo', 'by', 'bya', 'byo', 'cy', 'cya', 'cyo', 'hafi', 'ibi', 'ibyo', 'icyo', 'iki',
           'imwe', 'iri', 'iyi', 'iyo', 'izi', 'izo', 'ka', 'ko', 'ku', 'kuri', 'kuva', 'kwa', 'maze', 'mu', 'muri',
           'na', 'naho','nawe', 'ngo', 'ni', 'niba', 'nk', 'nka', 'no', 'nta', 'nuko', 'rero', 'rw', 'rwa', 'rwo', 'ry',
           'rya','ubu', 'ubwo', 'uko', 'undi', 'uri', 'uwo', 'uyu', 'wa', 'wari', 'we', 'wo', 'ya', 'yabo', 'yari', 'ye',
           'yo', 'yose', 'za', 'zo'}

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