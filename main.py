import torch.nn as nn
import torch
import argparse, os
from utils import Vectorizer, SentenceAnnotatorDataset
from model import Annotator
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter


class Config:
    relative_train_path = '/data/small_train.dat'
    relative_dev_path = '/data/small_valid.dat'
    min_freq = 1
    emsize = 512
    batch_size = 2
    lr = 0.0001
    log_interval = 5
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
writer = SummaryWriter(
    "runs/{}_{}_{}_{}_{}_{}".format(config.d1, config.d2, config.lr, config.emsize, config.batch_size,
                                    config.relative_train_path))
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

model = Annotator(len(vectorizer.word2idx), config.emsize, 4, config, args.cuda)

criterion = nn.NLLLoss()
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=config.lr)


def evaluate(dataset, model):
    valid_loader = DataLoader(dataset, config.batch_size)
    model.eval()
    loss = 0
    correct_predictions = 0
    tot_sentences = 0
    num_examples = 0
    for i, (abstracts, sentence_labels, num_of_sentences) in enumerate(valid_loader):
        output = model(abstracts)
        sentence_labels = sentence_labels.squeeze(2)
        correct_predictions += torch.sum(output.topk(1, dim=1)[1].squeeze(1) == sentence_labels).data
        loss += (criterion(output, sentence_labels) * abstracts.shape[0])
        num_examples += abstracts.shape[0]
        tot_sentences += (abstracts.shape[1] * abstracts.shape[0])

    return loss / num_examples, (correct_predictions * 100.) / tot_sentences


def train_epoches(dataset, model, n_epochs):
    train_loader = DataLoader(dataset, config.batch_size)
    for epoch in range(1, n_epochs + 1):
        per_epoch_loss = 0
        correct_predictions = 0
        tot_sentences = 0
        model.train(True)
        examples_processed = 0
        for i, (abstracts, sentence_labels, num_of_sentences) in enumerate(train_loader):
            output = model(abstracts)
            sentence_labels = sentence_labels.squeeze(2)
            correct_predictions += torch.sum(output.topk(1, dim=1)[1].squeeze(1) == sentence_labels).data
            loss = criterion(output, sentence_labels)

            model.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            examples_processed += abstracts.shape[0]
            tot_sentences += (abstracts.shape[1] * abstracts.shape[0])
            per_epoch_loss += (loss * abstracts.shape[0])

            if examples_processed % config.log_interval == 0:
                print("Epoch {}, examples processed {}, loss {}".format(epoch, examples_processed, loss))

        print(examples_processed)
        valid_loss, valid_accuracy = evaluate(annotator_valid_dataset, model)
        training_loss = per_epoch_loss / examples_processed
        training_accuracy = (correct_predictions * 100.) / tot_sentences
        writer.add_scalar("loss/training_loss", training_loss, epoch)
        writer.add_scalar("loss/valid_loss", valid_loss, epoch)
        print("Epoch {} complete \n --> Training Loss = {} \n --> Validation Loss = {} \
              \n --> Training Accuracy = {}% \n --> Validation Accuracy = {}% \n".format(epoch, training_loss, valid_loss, training_accuracy, valid_accuracy))


train_epoches(annotator_train_dataset, model, 20)