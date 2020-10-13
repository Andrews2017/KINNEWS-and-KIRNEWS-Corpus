from gensim.models import Word2Vec
import pandas as pd

stopset_kin = {'aba', 'abo', 'aha', 'aho', 'ari', 'ati', 'aya', 'ayo', 'ba', 'baba', 'babo', 'bari', 'be', 'bo', 'bose',
           'bw', 'bwa', 'bwo', 'by', 'bya', 'byo', 'cy', 'cya', 'cyo', 'dr', 'hafi', 'ibi', 'ibyo', 'icyo', 'iki',
           'imwe', 'iri', 'iyi', 'iyo', 'izi', 'izo', 'ka', 'ko', 'ku', 'kuri', 'kuva', 'kwa', 'maze', 'mu', 'muri',
           'na', 'naho','nawe', 'ngo', 'ni', 'niba', 'nk', 'nka', 'no', 'nta', 'nuko', 'rero', 'rw', 'rwa', 'rwo', 'ry',
           'rya','ubu', 'ubwo', 'uko', 'undi', 'uri', 'uwo', 'uyu', 'wa', 'wari', 'we', 'wo', 'ya', 'yabo', 'yari', 'ye',
           'yo', 'yose', 'za', 'zo'}

# load the data
data_train = pd.read_csv('../data/KINNEWS/cleaned/train.csv')
data_test = pd.read_csv('../data/KINNEWS/cleaned/test.csv')
data = pd.concat([data_train, data_test])
data['whole_doc'] = data['title'] + ' ' + data['content'].astype(str)

# clean the data (preprocessing)
data.whole_doc = data.whole_doc.str.replace('[^A-Za-z\s\’\-]+', '')
data.whole_doc = data.whole_doc.str.replace('[\n]+', '')
data.whole_doc = data.whole_doc.str.replace('^https?:\/\/.*[\r\n]*', '')

# Create the list of list format of the custom corpus for gensim modeling
sent = [row.split(' ') for row in data['whole_doc'] if len(row)]
sent = [[tok.lower() for tok in sub_sent if len(tok) != 0] for sub_sent in sent]
sent = [[tok.lower() for tok in sub_sent if tok not in stopset_kin] for sub_sent in sent]

# Training the model
w2v_model = Word2Vec(sent, window=5, min_count=5, sg=1, hs=1, size=100)

# Generate a list of words with their vectors to make the custom embeddings generation possible
w2v_vectors = []
for token, idx in w2v_model.wv.vocab.items():
    str_vec = ''
    if token in w2v_model.wv.vocab.keys():
        str_vec += token
        for i in range(len(w2v_model[token])):
            str_vec += ' ' + str(w2v_model[token][i])
    w2v_vectors.append(str_vec)

# Save the above embeddings list in txt file
with open("../pre-trained_embeddings/Kinyarwanda/W2V-Kin-50.txt", 'w') as output:
    for row in w2v_vectors:
        output.write(str(row) + '\n')