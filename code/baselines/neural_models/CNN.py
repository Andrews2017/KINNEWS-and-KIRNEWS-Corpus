import torch
from torchtext import data
import random
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
import torch.nn.functional as F

stopset = {'aba', 'abo', 'aha', 'aho', 'ari', 'ati', 'aya', 'ayo', 'ba', 'baba', 'babo', 'bari', 'be', 'bo', 'bose',
           'bw', 'bwa', 'bwo', 'by', 'bya', 'byo', 'cy', 'cya', 'cyo', 'dr', 'hafi', 'ibi', 'ibyo', 'icyo', 'iki',
           'imwe', 'iri', 'iyi', 'iyo', 'izi', 'izo', 'ka', 'ko', 'ku', 'kuri', 'kuva', 'kwa', 'maze', 'mu', 'muri',
           'na', 'naho','nawe', 'ngo', 'ni', 'niba', 'nk', 'nka', 'no', 'nta', 'nuko', 'rero', 'rw', 'rwa', 'rwo', 'ry',
           'rya','ubu', 'ubwo', 'uko', 'undi', 'uri', 'uwo', 'uyu', 'wa', 'wari', 'we', 'wo', 'ya', 'yabo', 'yari', 'ye',
           'yo', 'yose', 'za', 'zo'}

# Generate the custom embeddings
custom_embeddings = vocab.Vectors(name='../pre-trained_embeddings/kinyarwanda/W2V-Kin-50.txt',
                                  cache='../pre-trained_embeddings/kinyarwanda',
                                  unk_init=torch.Tensor.normal_)

### Prepare the data
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

## Load the dataset
# Define fields to hold the data
LABEL = data.LabelField(dtype=torch.float)
TITLE = data.Field(tokenize='spacy')
TEXT = data.Field(tokenize='spacy')

fields = [('label', LABEL), ('title', TITLE), ('content', TEXT)]

# Upload CSV format dataset and do train/test splits
train_data, test_data = data.TabularDataset.splits(
    path='../data/KINNEWS/cleaned',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=fields,
    skip_header=True  # dataset has a header(title)
)

# train/validation set split
train_data, valid_data = train_data.split(split_ratio=0.9, random_state=random.seed(SEED), )

# build the vocabulary
TEXT.build_vocab(train_data.title, train_data.content, max_size=15000, vectors=custom_embeddings)
TITLE.vocab = TEXT.vocab
LABEL.build_vocab(train_data)

# create the iterator and place the tensor it returned on GPU(if it is available)
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(TEXT.vocab.vectors)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.content),
    device=device)


# Build the model
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters,
                 filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# create the instance of the model
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 50
N_FILTERS = 150
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def multiclass_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.max(preds, -1)[1]
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# train the model
def train(model, iterator, optimizer, criterion):
    """ Training the model"""
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(torch.cat((batch.title, batch.content), 0)).squeeze(1)
        loss = criterion(predictions, batch.label.type(torch.cuda.LongTensor))
        acc = multiclass_accuracy(predictions, batch.label.type(torch.cuda.LongTensor))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# evaluate the model
def evaluate(model, iterator, criterion):
    """ Evaluating the model"""
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(torch.cat((batch.title, batch.content), 0)).squeeze(1)
            loss = criterion(predictions, batch.label.type(torch.cuda.LongTensor))
            acc = multiclass_accuracy(predictions, batch.label.type(torch.cuda.LongTensor))
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    N_EPOCHS = 8

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        print(f'\n| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% |'
              f' Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% |')

    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% |')

    print(f'The model has {count_parameters(model):,} trainable parameters')