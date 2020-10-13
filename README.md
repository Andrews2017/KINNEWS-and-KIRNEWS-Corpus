# KINNEWS-and-KIRNEWS
Data, Embeddings, Stopword lists, and code for [COLING 2020](https://coling2020.org/) paper titled "KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi" by [Rubungo Andre Niyongabo](https://scholar.google.com/citations?user=5qnTWQEAAAAJ&hl=en), [Hong Qu](https://scholar.google.com/citations?user=Aiq9mFMAAAAJ&hl=en), [Julia Kreutzer](https://scholar.google.co.uk/citations?user=j4cOSzAAAAAJ&hl=en), and Li Hunag.

## Data
### Download the datasets
- The raw and cleaned versions of KINNEWS can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1zxn0hgrOLlUsK5V0c7l71eAj1t2jiyox)
- The raw and cleaned versions of KIRNEWS can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1WNA5e_VRb4Jifgbvfvq4eCrQSDyd5Z_g)

### Datasets description
Each dataset is in camma-separated-value (csv) format, with columns that are described bellow:
| Field | Description |
| ----- | ----------- |
| label | Numerical labes that range from 1 to 14 |
| en_label | English labels |
| kin_label | Kinyarwanda labels |
| kir_label | Kirundi labels |
| url | The link to the news source |
| title | The title of the news article |
| content | The full content of the news article |

## Word embeddings
### Download pre-trained word emmbeddings
- The Kinyarwanda embeddings can be downloaded form [here](https://drive.google.com/drive/u/0/folders/1d-ZoTGErWLWAFdTAC8h9QRC3ZLByV0Zf)
- The Kirundi embeddings can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1tZPPAgp7UnciQxaDqs0haxPpgbJv1iZ1)

### Training your own embeddings 
To train you own word vectors, check [code/embeddings/word2vec_training.py](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS/tree/main/code/embeddings) file or refer to this [gensim](https://radimrehurek.com/gensim/models/word2vec.html) documentation.

## Stopwords

## Leaderboard (baselines)
### KINNEWS
| Model | Accuracy(%)|
| ----- | ----------- |
| BiGRU(W2V-Kin-50*) | 88.65 |
| SVM(TF-IDF) | 88.53 |
| BiGRU(W2V-Kin-100) | 88.29 |
| CNN(W2V-Kin-50) | 87.55 |
| CNN(W2V-Kin-100) | 87.54 |
| LR(TF-IDF) | 87.14 |
| MNB(TF-IDF) | 82.70 |
| Char-CNN | 71.70 |
### KIRNEWS
| Model | Accuracy(%)|
| ----- | ----------- |
| SVM(TF-IDF) | 90.14 |
| CNN(W2V-Kin-100) | 88.01 |
| BiGRU(W2V-Kin-100) | 86.61 |
| LR(TF-IDF) | 86.13|
| BiGRU(W2V-Kin-50) | 85.86 |
| CNN(W2V-Kin-50) | 85.75 |
| MNB(TF-IDF) | 82.67 |
| Char-CNN | 69.23 |

### Cross-lingual
| Model | Train set| Test set | Accuracy(%) |
| ----- | ----------- | ------- | ---------|
| MNB(TF-IDF) | KINNEWS | KIRNEWS | 73.46 |
| SVM(TF-IDF) | KINNEWS | KIRNEWS | 72.70 |
| LR(TF-IDF) | KINNEWS | KIRNEWS | 68.26 |
| BiGRU(W2V-Kin-50) | KINNEWS | KIRNEWS | 67.54 |
| BiGRU(W2V-Kin-100*) | KINNEWS | KIRNEWS | 65.06 |
| CNN(W2V-Kin-100) | KINNEWS | KIRNEWS | 61.72 |
| CNN(W2V-Kin-50) | KINNEWS | KIRNEWS | 60.64 |
| Char-CNN | KINNEWS | KIRNEWS | 49.60 |

| Model | Train set| Test set | Accuracy(%) |
| ----- | ----------- | ------- | ---------|
| CNN(W2V-Kin-100) | KIRNEWS | KIRNEWS | 88.01 |
| BiGRU(W2V-Kin-100) | KIRNEWS | KIRNEWS | 86.61 |
| CNN(W2V-Kin-50) | KIRNEWS | KIRNEWS | 85.75 |
| BiGRU(W2V-Kin-50) | KIRNEWS | KIRNEWS | 83.38 |
