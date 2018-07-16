from gensim.models import KeyedVectors
import json
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
import re

class Vectorizer:
    def __init__(self, max_words=None, min_frequency=None, start_end_tokens=True, maxlen=None):
        self.vocabulary = None
        self.vocabulary_size = 0
        self.word2idx = dict()
        self.idx2word = dict()
        self.max_words = max_words
        self.min_frequency = min_frequency
        self.start_end_tokens = start_end_tokens
        self.maxlen = maxlen

    def _find_max_short_text_length(self, corpus):
        self.maxlen = max(len(sent) for document in corpus for sent in document)
        if self.start_end_tokens:
            self.maxlen += 2

    def _build_vocabulary(self, corpus):
        vocabulary = Counter(word for document in corpus for sent in document for word in sent)
        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary) + 2  # padding and unk tokens
        if self.start_end_tokens:
            self.vocabulary_size += 2

    def _find_max_sentence_length(self, corpus):
        self.maxlen = max(len(sent) for document in corpus for sent in document)
        if self.start_end_tokens:
            self.maxlen += 2

    def fit(self, corpus):
        if not self.maxlen:
            self._find_max_sentence_length(corpus)
        self._build_vocabulary(corpus)
        self._build_word_index()

    def _build_word_index(self):
        self.word2idx['<UNK>'] = 3
        self.word2idx['<PAD>'] = 0
        self.word2idx['introduction'] = 4
        self.word2idx['body'] = 5
        self.word2idx['conclusion'] = 6

        if self.start_end_tokens:
            self.word2idx['<EOS>'] = 1
            self.word2idx['<SOS>'] = 2

        offset = len(self.word2idx)
        for idx, word in enumerate(self.vocabulary):
            self.word2idx[word] = idx + offset
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def add_start_end(self, vector):
        vector.append(self.word2idx['<EOS>'])
        return [self.word2idx['<SOS>']] + vector

    def transform_sentence(self, sentence, start_end_tokens=True):
        """
        Vectorize a single sentence
        """
        vector = [self.word2idx.get(word, 3) for word in sentence]
        if start_end_tokens:
            vector = self.add_start_end(vector)
        return vector

    def transform(self, corpus, start_end_tokens=True):
        """
        Vectorizes a corpus in the form of a list of lists.
        A corpus is a list of documents and a document is a list of sentence.
        """
        vcorpus = []
        for document in corpus:
            vcorpus.append([self.transform_sentence(sentence, start_end_tokens) for sentence in document])
        return vcorpus

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

class SentenceAnnotatorDataset(Dataset):
    def __init__(self, path, vectorizer, use_cuda, max_len=200, max_sent=10):
        self.max_number_of_sentences = -1
        self.vectorizer = vectorizer
        self.max_len = max_len
        self.max_sent = max_sent
        self.abstracts, self.labels = self.process(path)
        self.cuda = use_cuda
        self.abstracts, self.labels = self._vectorize_corpus()
        self._initial_corpus()

    def process(self, path):
        abstracts, sent_labels = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                j = json.loads(line)
                sentences = j["sents"]
                labels = j["labels"]
                S, L = [self._tokenize_word(s) for s in sentences], [[l] for l in labels]
                if len(S) > self.max_sent:
                    continue
                self.max_number_of_sentences = max(self.max_number_of_sentences, len(S))
                abstracts.append(S)
                sent_labels.append(L)
            return abstracts, sent_labels

    def _pad_sentence_vector(self, vector, maxlen, pad_value=0):
        org_length = len(vector)
        padding = maxlen - org_length
        vector.extend([pad_value] * padding)
        return vector

    def _pad_abstract(self, abstract, sent_length):
        padded_abstract = []
        for sent in abstract:
            padded_abstract.append(self._pad_sentence_vector(sent, sent_length))

        if len(abstract) < self.max_number_of_sentences:
            for _ in range(self.max_number_of_sentences - len(abstract)):
                padded_abstract.append(self._pad_sentence_vector([], sent_length))

        return padded_abstract

    def _initial_corpus(self):
        old = []
        max_sentence_length = -1
        for ab, le in zip(self.abstracts, self.labels):
            for sent in ab:
                max_sentence_length = max(max_sentence_length, len(sent))
            old.append((ab, le))

        old.sort(key = lambda x: len(x[0]), reverse = True)
        self.abstracts, self.labels = [], []
        for abstract, structure_labels in old:
            self.abstracts.append((self._pad_abstract(abstract, max_sentence_length), [len(abstract)]))
            self.labels.append(self._pad_abstract(structure_labels, 1))

    def _vectorize_corpus(self):
        if not self.vectorizer.vocabulary:
            self.vectorizer.fit(self.abstracts)
        return self.vectorizer.transform(self.abstracts, start_end_tokens=True), self.vectorizer.transform(self.labels, start_end_tokens=False)

    def _tokenize_word(self, sentence):
        sentence = self.vectorizer.clean_str(sentence)
        result = []
        for word in sentence.split():
            if word:
                result.append(word)
        return result


    def __getitem__(self, index):
        abstracts, num_of_sentences = self.abstracts[index]
        sentence_labels = self.labels[index]

        abstracts = torch.LongTensor(abstracts)
        sentence_labels = torch.FloatTensor(sentence_labels)
        num_of_sentences = torch.LongTensor(num_of_sentences).view(-1, 1)

        if self.cuda:
            abstracts = abstracts.cuda()
            sentence_labels = sentence_labels.cuda()
            num_of_sentences = num_of_sentences.cuda()

        return abstracts, sentence_labels, num_of_sentences

    def __len__(self):
        return len(self.abstracts)

#provide pretrained embeddings for text
def load_embeddings(pytorch_embedding, word2idx, filename, embedding_size):
    print("Copying pretrained word embeddings from ", filename, flush=True)
    en_model = KeyedVectors.load_word2vec_format(filename)
    """ Fetching all of the words in the vocabulary. """
    pretrained_words = set()
    for word in en_model.vocab:
        pretrained_words.add(word)

    arr = [0] * len(word2idx)
    for word in word2idx:
        index = word2idx[word]
        if word in pretrained_words:
            arr[index] = en_model[word]
        else:
            arr[index] = np.random.uniform(-1.0, 1.0, embedding_size)

    """ Creating a numpy dictionary for the index -> embedding mapping """
    arr = np.array(arr)
    """ Add the word embeddings to the empty PyTorch Embedding object """
    pytorch_embedding.weight.data.copy_(torch.from_numpy(arr))
    return pytorch_embedding