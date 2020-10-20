# KINNEWS-and-KIRNEWS
Data, Embeddings, Stopword lists, code, and baselines for [COLING 2020](https://coling2020.org/) paper titled "KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi" by [Rubungo Andre Niyongabo](https://scholar.google.com/citations?user=5qnTWQEAAAAJ&hl=en), [Hong Qu](https://scholar.google.com/citations?user=Aiq9mFMAAAAJ&hl=en), [Julia Kreutzer](https://scholar.google.co.uk/citations?user=j4cOSzAAAAAJ&hl=en), and Li Huang.

**Note:** Please, when using any of the resources provided here, remember to cite our paper.

## Data
### Download the datasets
- The raw and cleaned versions of KINNEWS can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1zxn0hgrOLlUsK5V0c7l71eAj1t2jiyox) (21,268 articles, 14 classes, 45.07MB(raw) and 38.85MB(cleaned))
- The raw and cleaned versions of KIRNEWS can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1WNA5e_VRb4Jifgbvfvq4eCrQSDyd5Z_g) (4,612 articles, 12 classes, 9.31MB(raw) and 7.77MB(cleaned))

### Datasets description
Each dataset is in camma-separated-value (csv) format, with columns that are described bellow:
| Field | Description |
| ----- | ----------- |
| label | Numerical labels that range from 1 to 14 |
| en_label | English labels |
| kin_label | Kinyarwanda labels |
| kir_label | Kirundi labels |
| url | The link to the news source |
| title | The title of the news article |
| content | The full content of the news article |

## Word embeddings
### Download pre-trained word emmbeddings
- The Kinyarwanda embeddings can be downloaded form [here](https://drive.google.com/drive/u/0/folders/1d-ZoTGErWLWAFdTAC8h9QRC3ZLByV0Zf) (59.88MB for 100d
 and 29.94MB for 50d))
- The Kirundi embeddings can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1tZPPAgp7UnciQxaDqs0haxPpgbJv1iZ1) (17.98MB for 100d and 8.96MB for 50d))

### Training your own embeddings 
To train you own word vectors, check [code/embeddings/word2vec_training.py](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS/tree/main/code/embeddings) file or refer to this [gensim](https://radimrehurek.com/gensim/models/word2vec.html) documentation.

## Stopwords
To use our stopwords you may just copy the whole [stopset_kin](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS/blob/main/stopwords/Kinyarwanda/Kinyarwanda%20stopwords%20set.txt) for Kinyarwanda and [stopset_kir](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS/blob/main/stopwords/Kirundi/Kirundi%20stopwords%20set.txt) for Kirundi into your code or import them directly from [KKLTK](https://github.com/Andrews2017/kkltk) package, which is more recommended.

## Leaderboard (baselines)
### Monolingual
#### KINNEWS
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
#### KIRNEWS
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
