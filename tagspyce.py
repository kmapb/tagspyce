# Train a simple model on a collection of hashtagged data.
import sys
import torch
from stringzilla import Str, File
import os

def dprint(str):
    if os.getenv('DBG'):
        print(str)

class WordDict:
    def __init__(self, max_words):
        self.word_2_idx = {}
        self.max_words = max_words
        self.num_words = 0

    def get(self, word):
        #dprint("get {}".format(str(word)))
        if word in self.word_2_idx:
            #dprint("hit {} -> {}".format(str(word), self.word_2_idx[word]))
            return self.word_2_idx[word]
        if self.num_words == self.max_words:
            dprint("capacity miss {}".format(str(word), self.word_2_idx[word]))
            return None
        dprint("fill miss {} -> {}".format(str(word), self.num_words))
        self.word_2_idx[word] = self.num_words
        self.num_words += 1
        assert self.has(word)
        return self.num_words - 1

    def has(self, word):
        return word in self.word_2_idx

    def encode(self, words):
        return torch.Tensor([self.get(word) for word in words], dtype=torch.long)        

class MarginLoss(torch.nn.Module):
    def __init__(self, margin=1e-1):
        torch.nn.Module.__init__(self)
        self.cosine = torch.nn.CosineSimilarity(dim=0)
        self.margin = margin

    def _sim(self, a, b):
        return self.cosine(a, b)

    def forward(self, xs, ys, ynegs):
        crude = self._sim(xs, ynegs) - self._sim(xs, ys) + self.margin
        if crude <= 0.0:
            return torch.tensor(0.0, device=crude.device, requires_grad=True)
        return crude

class WordModel(torch.nn.Module):
    def __init__(self, dict_size, emb_size, margin=1e-1):
        torch.nn.Module.__init__(self)
        self.dict_size = dict_size
        self.emb_size = emb_size 
        self.word_embs = torch.nn.Embedding(dict_size, emb_size)
        self.tag_embs = torch.nn.Embedding(dict_size, emb_size)
        self.margin = MarginLoss(margin)

    def _search_violators(xs, sim, max_probes):
        indices = torch.randint(0, self.dict_size, (max_probes,))
        ys = self.word_embs.forward(indices)
        sims = self._sim(xs, ys)

    def forward(self, idx, targets_pos, targets_neg):
        # Get a sparse set of vectors for keys, targets
        xs = self.word_embs.forward(idx)
        ys = self.tag_embs.forward(targets_pos)
        negs = self.tag_embs.forward(targets_neg)

        # Collapse them down to one vector
        xs = torch.einsum('ij->j', xs)
        ys = torch.einsum('ij->j', ys)
        negs = torch.einsum('ij->j', negs)

        # Compute the similarity between them
        assert xs.shape == (self.emb_size,)
        assert ys.shape == (self.emb_size,)
        assert negs.shape == (self.emb_size,)

        return self.margin(xs, ys, negs)

def project(wd, words):
    first_pass = [ wd.get(word) for word in words]
    f = [ f for f in filter(lambda x: x is not None, first_pass) ]
    return torch.LongTensor(f)

def main(argv):
    word_dict_len = int(1e5)
    word_dim = 128

    text = Str(File(argv[1]))

    wd = WordDict(word_dict_len)
    wm = WordModel(word_dict_len, word_dim)
    negatives = 0
    positives = 0
    lines = 0
    words = 0

    print("training!")
    optim = torch.optim.Adam(wm.parameters(), lr=1e-3)
    wm.train()
    example = 0
    for line in text.splitlines():
        optim.zero_grad()
        words = line.split()
        xs = project(wd, words[:-1])
        ys = project(wd, words[-1:])
        negs = torch.randint(0, word_dict_len, size=(120,))
        loss = wm.forward(xs, ys, negs)
        example += 1
        if (example % 100) == 1:
            print("loss {}".format(loss))
        loss.backward()
        optim.step()

if __name__ == '__main__':
    main(sys.argv)
