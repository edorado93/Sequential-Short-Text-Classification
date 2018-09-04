import torch.nn as nn
import torch
import argparse, os
from utils import Vectorizer, SentenceAnnotatorDataset
from model import LSTMAnnotator, CNNAnnotator
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import json
import configurations
import utils


parser = argparse.ArgumentParser(description='seq2seq model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--conf', type=str,
                    help="configuration to load for the training")
parser.add_argument('--mode', type=int,
                    help='train = 0, eval = 1', default=0)
parser.add_argument('--arch', type=str,
                    help="Which architecture to use. lstm or conv")

args = parser.parse_args()
config = configurations.get_conf(args.conf)
save = "eval={}_hidden={}_d1={}_d2={}_lr={}_emsize={}_dropout={}_weight-decay={}_pretrained={}_arch={}.pkl".format(config.eval_using, config.hidden_size, config.d1, config.d2, config.lr, config.emsize, config.input_dropout, config.weight_decay, config.pretrained is not None, args.arch)
writer = SummaryWriter("runs/"+save)
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

# We don't want to change the size of word embeddings. The additional dimension comes from
# the attention scores. Weighted attention is where we will do an element wise dot product of
# the attention scores with the word embeddings before feeding to the LSTM.
embedding = nn.Embedding(len(vectorizer.word2idx), config.emsize)
if config.use_attention and not config.attention_type == "weighted":
    config.emsize += 1

if config.pretrained:
    embedding = utils.load_embeddings(embedding, vectorizer.word2idx, config.pretrained, config.emsize)

if args.arch == "lstm":
    model = LSTMAnnotator(embedding, config.emsize, config.hidden_size, config.input_dropout, 4, config, args.cuda)
else:
    model = CNNAnnotator(embedding, config.emsize, config.hidden_size, config.input_dropout, 4, config, args.cuda)
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Model total parameters:', total_params, flush=True)
print("Configuration: Learning Rate = {}, Embedding dimension = {} Hidden Units = {},\
      \n FF1 = {}, FF2 = {}, Dropout Probability = {} \
      \n Pretrained embeddings = {}, Architecture = {}".format(config.lr, config.emsize, config.hidden_size,
                                                               config.d1, config.d2,
                                                               config.input_dropout, config.pretrained,
                                                               args.arch))

criterion = nn.NLLLoss()
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def evaluate(dataset, model):
    valid_loader = DataLoader(dataset, config.batch_size)
    model.eval()
    loss = 0
    correct_predictions = 0
    tot_sentences = 0
    for i, (abstracts, sentence_labels, num_of_sentences) in enumerate(valid_loader):
        output = model(abstracts)
        sentence_labels = sentence_labels.squeeze(2)
        correct_predictions += torch.sum(output.topk(1, dim=1)[1].squeeze(1) == sentence_labels).data
        sentences_processed = abstracts.shape[1] * abstracts.shape[0]
        loss += criterion(output, sentence_labels).item() * sentences_processed
        tot_sentences += sentences_processed

    return loss / tot_sentences, (correct_predictions * 100.) / tot_sentences


def train_epoches(dataset, model, n_epochs):
    train_loader = DataLoader(dataset, config.batch_size)
    best_loss = 100.
    best_accuracy = 0.
    patience = config.patience
    for epoch in range(1, n_epochs + 1):
        interval_loss = 0
        interval_sentences = 0
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

            sentences_processed = abstracts.shape[1] * abstracts.shape[0]
            examples_processed += abstracts.shape[0]
            tot_sentences += sentences_processed
            interval_sentences += sentences_processed
            per_epoch_loss += loss.item() * sentences_processed
            interval_loss += loss.item() * sentences_processed

            if examples_processed % config.log_interval == 0:
                print("Epoch {}, examples processed {}, loss {:.4f}".format(epoch, examples_processed, interval_loss / interval_sentences))
                interval_loss = 0
                interval_sentences = 0

        valid_loss, valid_accuracy = evaluate(annotator_valid_dataset, model)
        training_loss = per_epoch_loss / tot_sentences
        training_accuracy = (correct_predictions * 100.) / tot_sentences
        writer.add_scalar("loss/training_loss", training_loss, epoch)
        writer.add_scalar("loss/valid_loss", valid_loss, epoch)
        print("Epoch {} complete \n --> Training Loss = {:.4f} \n --> Validation Loss = {:.4f} \
              \n --> Training Accuracy = {}% \n --> Validation Accuracy = {}% \n".format(epoch, training_loss, valid_loss, training_accuracy, valid_accuracy))

        if config.eval_using == "loss" and valid_loss < best_loss:
            best_loss = valid_loss
            print("Saving model with best loss till now")
            torch.save(model.state_dict(), save)
            patience = config.patience
        elif config.eval_using == "accuracy" and valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            print("Saving model with best accuracy till now")
            torch.save(model.state_dict(), save)
            patience = config.patience
        else:
            patience -= 1
            if patience == 0:
                break


if __name__ == "__main__":
    if args.mode == 0:
        train_epoches(annotator_train_dataset, model, config.epochs)
    if args.mode == 1:
        model.load_state_dict(torch.load(save))
        test_data_path = cwd + config.relative_test_path
        test_dataset = SentenceAnnotatorDataset(training_data_path, vectorizer, args.cuda, max_len=1000)
        test_loader = DataLoader(test_dataset, config.batch_size)

        predictions = []
        with open("predictions.txt", "w") as f:
            for i, (abstracts, sentence_labels, num_of_sentences) in enumerate(test_loader):
                output = model(abstracts)
                pred = output.topk(1, dim=1)[1].squeeze(1)

                for a, p in zip(abstracts, pred):
                    sents, labels = [], []
                    for s1, s2 in zip(a, p):
                        # Don't consider padded sentences in the end. The extra ones
                        if s1[0].item() != 3:
                            sents.append(" ".join([vectorizer.idx2word[w.item()] for w in s1 if w.item() not in [3,5,6]]))
                            labels.append(vectorizer.idx2word[s2.item()])

                    j = {"sents": sents, "labels": labels}
                    f.write(json.dumps(j)+"\n")

