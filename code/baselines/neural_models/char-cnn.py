import os
import sys
import csv
import torch
import shutil
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
csv.field_size_limit(sys.maxsize)

class MyDataset(Dataset):
    def __init__(self, data_path, max_length=1500):
        self.data_path = data_path
        self.vocabulary = list("""abcdefghijklmnoprstuvwyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
              if idx != 0:
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                label = line[0]
                texts.append(text)
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

class CharacterLevelCNN(nn.Module):
    def __init__(self, n_classes=14, input_length=1500, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        # compute the  output shape after forwarding an input to the conv layers
        input_shape = (128,
                      input_length,
                      input_dim)
        self.output_dimension = self._get_conv_output(input_shape)

        self.fc1 = nn.Sequential(nn.Linear(self.output_dimension, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    def forward(self, input):
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

def train(feature,log_path, optimizer):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if not os.path.exists(output):
        os.makedirs(output)
    output_file = open(output + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format())

    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    test_params = {"batch_size": batch_size,
                   "shuffle": True,
                   "num_workers": 0}
    training_set = MyDataset(input + "/train.csv", max_length)
    test_set = MyDataset(input + "/test.csv", max_length)
    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)

    if feature == "small":
        model = CharacterLevelCNN(input_length=max_length, n_classes=training_set.num_classes,
                                  input_dim=len(alphabet),
                                  n_conv_filters=256, n_fc_neurons=1024)

    elif feature == "large":
        model = CharacterLevelCNN(input_length=max_length, n_classes=training_set.num_classes,
                                  input_dim=len(alphabet),
                                  n_conv_filters=1024, n_fc_neurons=2048)
    else:
        sys.exit("Invalid feature mode!")

    log_path = "{}_{}_{}".format(log_path, feature, dataset)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    for epoch in range(num_epochs):
        for iter, batch in enumerate(training_generator):
            features, label = batch
            label = np.array(label, int)
            label = torch.Tensor(label)
            if torch.cuda.is_available():
                features = features.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, label.type(torch.cuda.LongTensor))
            loss.backward()
            optimizer.step()

            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy"])

            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        print("Epoch: {}/{}, Lr: {}, Train Loss: {}, Train Accuracy: {}".format(
                epoch + 1,
                num_epochs,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))

        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for batch in test_generator:
            te_feature, te_label = batch
            num_sample = len(te_label)
            te_label = np.array(te_label, int)
            te_label = torch.Tensor(te_label)
            if torch.cuda.is_available():
                te_feature = te_feature.cuda()
                te_label = te_label.cuda()
            with torch.no_grad():
                te_predictions = model(te_feature)
            te_loss = criterion(te_predictions, te_label.type(torch.cuda.LongTensor))
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())

        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
        output_file.write(
            "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, num_epochs,
                te_loss,
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        print("Epoch: {}/{}, Lr: {}, Test Loss: {}, Test Accuracy: {}".format(
            epoch + 1,
            num_epochs,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        writer.add_scalar('Test/Loss', te_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        model.train()
        if te_loss + es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model, "{}/char-cnn_{}_{}".format(output, dataset, feature))
        # Early stopping
        if epoch - best_epoch > es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, te_loss, best_epoch))
            break
        if optimizer == "sgd" and epoch % 3 == 0 and epoch > 0:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True

    alphabet = "abcdefghijklmnoprstuvwyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}"
    max_length = 1500
    optimizer = "sgd"
    batch_size = 128
    num_epochs = 20
    lr = 0.001
    es_min_delta = 0.0
    es_patience = 3
    input = "../data/KINNEWS/cleaned"
    output = "../output"
    log_path = "../tensorboard/char-cnn"
    train("small", log_path, optimizer)
