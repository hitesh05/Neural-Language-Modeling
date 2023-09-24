import torch
import torch.nn as nn
from torch import tensor
from torchtext import vocab
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from elmo_finetune import Elmo_classifier
import sklearn 
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ProcessDataset():
    def __init__(self, filepath, min_freq=1):
        self.specials = ['<UNK>', '<PAD>']
        df = pd.read_csv(filepath)
        all_words = list()
        for i in tqdm(range(len(df)), desc="building vocabulary"):
            if i < 7600:
                continue
            if i > 35000:
                break
            line = df['Description'][i]
            l = [[word.lower()] for word in word_tokenize(line)]
            all_words.extend(l)

        self.vocab = vocab.build_vocab_from_iterator(all_words, min_freq=min_freq, specials=self.specials)
        def_idx = self.vocab[self.specials[0]]
        self.vocab.set_default_index(def_idx)

class ClassificationDataset(Dataset):
    def __init__(self,vocabulary, filename, is_train=True):
        self.vocab = vocabulary
        self.tokens = list()
        self.tokens_f = list()
        self.tokens_b = list()
        self.labels = list()
        self.max_len = -1
        self.df = pd.read_csv(filename)
        self.specials = ['<UNK>', '<PAD>']
        self.is_train = is_train
        self.make_dataset()

    def make_dataset(self):
        for i in tqdm(range(len(self.df)), desc='tokenising'):
            # first 7600 sents of train as dev
            # only take around 34k sents for train to avoid running out of memory
            # take test = dev set = 7600 sents
            if self.is_train:
                if i < 7600:
                    continue
                if i > 8600:
                    break
            else:
                if i > 7600:
                    continue

            # iterate over the dataset
            line = self.df['Description'][i]
            label = self.df['Class Index'][i]
            self.labels.append(label)
            words = list()
            indices = list()
            for word in word_tokenize(line):
                words.append(word.lower())
            for word in words:
                indices.append(self.vocab[word])

            '''
            making context for fwd and bwd lstm
            only consider the complete sentence instead of
            profressively increasing the n-grams since we are
            not training an lstm, and only have to pass the
            inputs to get ans

            maintaining fwd and bwd again because we have to
            pass the sentence to the elmo pretrained model to
            get the contextualised embeddings
            '''
            self.tokens.append(indices)
            # self.tokens_f = self.tokens.copy()
            indices.reverse()
            self.tokens_b.append(indices)

            x = len(indices)
            self.max_len = x if x>self.max_len else self.max_len

        self.tokens = [t + [self.vocab[self.specials[1]]]*(self.max_len-len(t)) for t in self.tokens]
        self.tokens_f = self.tokens.copy()
        self.tokens_b = [t + [self.vocab[self.specials[1]]]*(self.max_len-len(t)) for t in self.tokens_b]
        self.labels = tensor(self.labels)
        self.tokens = tensor(self.tokens)
        self.tokens_f = tensor(self.tokens_f)
        self.tokens_b = tensor(self.tokens_b)

    def __getitem__(self, idx):
        return self.tokens[idx].to(device), self.tokens_f[idx].to(device), self.tokens_b[idx].to(device), self.labels[idx].to(device)
    def __len__(self):
        return len(self.labels)

def get_embeddings(vocabulary):
    def get_unk(v):
        return torch.mean(v, dim=0)
    em_name = '6B'
    dim = 300
    glove = vocab.GloVe(name=em_name, dim=dim)
    unk_emb = get_unk(glove.vectors)
    embeds = list()
    for word in tqdm(vocabulary.get_itos(), desc='getting glove embeddings'):
        if word in glove.itos:
            embeds.append(glove[word])
        else:
            embeds.append(unk_emb)
    embeds = torch.stack(embeds)
    return embeds   
    
preprocessed_data = ProcessDataset("elmo_data/train.csv")
cls_dataset = ClassificationDataset(preprocessed_data.vocab, "elmo_data/train.csv")
cls_dataset_test = ClassificationDataset(preprocessed_data.vocab, "elmo_data/test.csv", is_train=False)
# elmo_dataset_val = ElmoDataset(preprocessed_data.vocab, 'elmo_data/train.csv')
embedding_matrix = get_embeddings(preprocessed_data.vocab)

mdl = Elmo_classifier('elmo_pretrained.pth', preprocessed_data.vocab, embedding_matrix).to(device)
# train_loader = DataLoader(cls_dataset, batch_size=32)
pad_idx = preprocessed_data.vocab['<PAD>']
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(mdl.parameters(), lr=1e-3)

num_epochs = 5

# training fn
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(cls_dataset):
        x, x_f ,x_b, label = data
        output = mdl(x, x_f, x_b)
        
        loss = loss_fn(output, label)
        # zero the parameter gradients
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(cls_dataset)
    running_loss = 0
    print(f"Average loss for epoch {epoch}: {avg_loss}")

torch.save(mdl.state_dict(), 'elmo_finetuned.pth')
# torch.save(mdl.state_dict(), 'elmo_finetuned1.pth')
# torch.save(mdl.state_dict(), 'elmo_finetuned2.pth')
# torch.save(mdl.state_dict(), 'elmo_finetuned3.pth')
# torch.save(mdl.state_dict(), 'elmo_finetuned4.pth')

# Test the model
def accuracy(dataset):
    mdl.eval()
    running_loss = 0
    correct_pred_b = 0

    total_b = 0
    y_pred,y_true=[],[]
    for i, data in enumerate(dataset):
        x_f,x_b,x,labels = data
        output=mdl(x,x_f,x_b)
        loss = loss_fn(output, labels)
        output = nn.Softmax(dim=0)(output)
        y_pred.append(torch.argmax(output).item())
        y_true.append(label.item())
        if torch.argmax(output) == label:
            correct_pred_b += 1
            
        total_b += 1
        running_loss+=loss.item()
    
    running_loss /= len(dataset)
    mdl.train()
    print("Loss: ", running_loss, "Accuracy", correct_pred_b / total_b)
    print("Multilabel Confusion Matrix: ",multilabel_confusion_matrix(y_true, y_pred))
    print("F1 score macro: ", f1_score(y_true, y_pred, average='macro'))
    print("F1 score micro: ", f1_score(y_true, y_pred, average='micro'))
    
accuracy(cls_dataset_test)