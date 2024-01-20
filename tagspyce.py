# Train a simple model on a collection of hashtagged data.
import sys
import torch
from stringzilla import Str, File
from datasets import load_dataset
import os
import sqlite3
from progress.bar import Bar

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
            return self.word_2_idx[word]
        if self.num_words == self.max_words:
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

class TagSpaceModel(torch.nn.Module):
    def __init__(self, dict_size, emb_size, margin=1e-1):
        torch.nn.Module.__init__(self)
        self.dict_size = dict_size
        self.emb_size = emb_size 
        self.word_embs = torch.nn.Embedding(dict_size, emb_size)
        self.tag_embs = torch.nn.Embedding(dict_size, emb_size)
        self.margin = MarginLoss(margin)

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
    try:
        tagless = list(filter(lambda s: len(s) > 0 and s[0] != '#', words))
        tokenized = [ wd.get(word) for word in tagless ]
        f = [ f for f in filter(lambda x: x is not None, tokenized) ]
        return torch.LongTensor(f)
    except:
        import pdb; pdb.set_trace()

def main(argv):
    word_dict_len = int(1e5)
    tag_dict_len = int(1e5)
    word_dim = 128

    wd = WordDict(word_dict_len)
    tagd = WordDict(tag_dict_len)
    model = TagSpaceModel(word_dict_len, word_dim)

    print("opening the DB!")
    conn = sqlite3.connect(sys.argv[1])
    cursor = conn.execute('SELECT count(1) from tweets;')
    count = cursor.fetchone()[0]

    cursor = conn.execute("SELECT * FROM tweets")
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    model = model.cuda()
    example = 0

    with Bar('training', max=count) as bar:
        for row in cursor:
            bar.next()
            assert len(row) == 3
            text = row[1]
            words = text.split()

            hashtag_str = row[2]
            hashtags = hashtag_str.split(',')
            xs = project(wd, words).cuda()
            ys = project(tagd, hashtags).cuda()

            optim.zero_grad()
            negs = torch.randint(0, tagd.num_words, size=(120,)).cuda()
            loss = model.forward(xs, ys, negs)
            example += 1
            if (example % 100) == 1:
                print("loss {} words {} tags {}".format(loss, wd.num_words, tagd.num_words))
            loss.backward()
            optim.step()
    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    main(sys.argv)
