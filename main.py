import torch.nn as nn
import torch
import argparse, os
from utils import Vectorizer, SentenceAnnotatorDataset
from model import Annotator
from torch.utils.data import DataLoader
import torch.optim as optim

class Config:
    relative_train_path = '/data/small.dat'
    relative_dev_path = '/data/small.dat'
    min_freq = 1
    emsize = 512
    batch_size = 2
    lr = 0.001
    log_interval = 200
    max_grad_norm = 10
    d1 = 0
    d2 = 0

parser = argparse.ArgumentParser(description='seq2seq model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--conf', type=str,
                    help="configuration to load for the training")

config = Config()
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(1111)

cwd = os.getcwd()
training_data_path = cwd + config.relative_train_path
validation_data_path = cwd + config.relative_dev_path

vectorizer = Vectorizer(min_frequency=config.min_freq)
annotator_train_dataset = SentenceAnnotatorDataset(training_data_path, vectorizer, args.cuda, max_len=1000)
annotator_valid_dataset = SentenceAnnotatorDataset(validation_data_path, vectorizer, args.cuda, max_len=1000)

model = Annotator(len(vectorizer.word2idx), config.emsize, 3, config, args.cuda)

criterion = nn.CrossEntropyLoss(ignore_index=0)
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

def train_epoches(dataset, model, n_epochs):
    train_loader = DataLoader(dataset, config.batch_size)
    for epoch in range(1, n_epochs + 1):
        model.train(True)
        examples_processed = 0
        for i, (abstracts, sentence_labels, num_of_sentences) in enumerate(train_loader):
                output = model(abstracts)
                loss = criterion(output, sentence_labels)

                model.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                examples_processed += abstracts.shape[0]

                if examples_processed % config.log_interval == 0:
                    print("Epoch {}, examples processed {}, loss {}".format(epoch, examples_processed, loss))

train_epoches(annotator_train_dataset, model, 100)