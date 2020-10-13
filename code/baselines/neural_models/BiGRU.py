import torch
from torchtext import data
import random
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab

stopset_kin = {'aba', 'abo', 'aha', 'aho', 'ari', 'ati', 'aya', 'ayo', 'ba', 'baba', 'babo', 'bari', 'be', 'bo', 'bose',
           'bw', 'bwa', 'bwo', 'by', 'bya', 'byo', 'cy', 'cya', 'cyo', 'dr', 'hafi', 'ibi', 'ibyo', 'icyo', 'iki',
           'imwe', 'iri', 'iyi', 'iyo', 'izi', 'izo', 'ka', 'ko', 'ku', 'kuri', 'kuva', 'kwa', 'maze', 'mu', 'muri',
           'na', 'naho','nawe', 'ngo', 'ni', 'niba', 'nk', 'nka', 'no', 'nta', 'nuko', 'rero', 'rw', 'rwa', 'rwo', 'ry',
           'rya','ubu', 'ubwo', 'uko', 'undi', 'uri', 'uwo', 'uyu', 'wa', 'wari', 'we', 'wo', 'ya', 'yabo', 'yari', 'ye',
           'yo', 'yose', 'za', 'zo'}

## Build the model
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
TITLE = data.Field(tokenize='spacy', stop_words=stopset_kin)
TEXT = data.Field(tokenize='spacy', stop_words=stopset_kin)

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

# Train/validation set split
train_data, valid_data = train_data.split(split_ratio=0.9, random_state=random.seed(SEED))

# Build the vocabulary
TEXT.build_vocab(train_data.title, train_data.content, max_size=10000, vectors=custom_embeddings)
TITLE.vocab = TEXT.vocab
LABEL.build_vocab(train_data)

# Create the iterator and place the tensor it returned on GPU(if it is available)
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.content),
    device=device)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden.squeeze(0))

# Create the instance of the model
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 50
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def multiclass_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # Round predictions to the closest integer
    rounded_preds = torch.max(preds, -1)[1]
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

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
    """ Counting the model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    N_EPOCHS = 10

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        print(f'\n| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% |'
              f' Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% |')

    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% |')

    print(f'The model has {count_parameters(model):,} trainable parameters')