# Train a simple model on a collection of hashtagged data.
import sys
import torch
from stringzilla import Str, File

class WordDict:
    def __init__(self, max_words):
        self.word_2_idx = {}
        self.max_words = max_words
        self.num_words = 0

    def get(self, word):
        if word in word_2_idx:
            return word_2_idx[word]
        if self.num_words == self.max_words:
            return None
        self.word_2_idx[word] = self.num_words
        self.num_words += 1

word_dict_len = int(1e5)
word_2_idx = {}
word_dim = 128

class WordModel(torch.nn.Module):
    def __init__(self, dict_size, emb_size, margin=1e-4):
        self.dict_size = dict_size
        self.emb_size = emb_size 
        self.word_embs = torch.randn((word_dict_len, word_dim))

    def _sim(xs, ys):
        return xs.dot(ys)

    def _search_violators(xs, sim, max_probes):
        indices = torch.randint(0, self.dict_size, (max_probes,))
        ys = self.word_embs.forward(indices)
        sims = self._sim(xs, ys)

    def forward(self, idx, targets):
        # Get a sparse set of vectors for keys, targets
        xs = self.word_embs.forward(idx)
        ys = self.word_embs.forward(ys)

        # Collapse them down to one vector
        xs = torch.einsum('i,j->i', xs)
        ys = torch.einsum('i,j->i', ys)

        # Compute the similarity between them
        assert xs.shape == (word_dim,)
        assert ys.shape == (word_dim,)

        return self._sim(xs, ys)

text = Str(File(sys.argv[1]))

wd = WordDict(word_dict_len)
wm = WordModel(word_dict_len, word_dim)
negatives = 0
positives = 0
lines = 0
words = 0

def project(wd, words):
    first_pass = [ wd.get(word) for word in words]
    return filter(lambda x: x is not None, first_pass)

# Stats!
for line in text.splitlines():
    lines += 1
    split = line.split()
    xs = project(wd, split[:-1])
    ys = project(wd, split[-1:])
    loss = wm.forward(xs, ys)
    print("loss {}".format(loss))
    for word in split:
        wid = wd.get(word)
        if wid == None:
            negatives += 1
        else:
            positives += 1
        words += 1
        if words % 1000 == 0:
            print("@ word {}: {} lines\n".format(words, lines))

print("negatives: {} ({}%) of {} dictwords".format(negatives,
                                               negatives / (negatives +
                                               positives),
                                               wd.num_words))

print("words: {}".format(words))
print("lines: {}".format(lines))
sys.exit()
print("training!")
for line in text.splitlines():
    xs = project(wd, words[:-1])
    ys = project(wd, words[-1:])
    loss = wm.forward(xs, ys)
    print("loss {}".format(loss))