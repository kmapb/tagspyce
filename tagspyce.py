# Train a simple model on a collection of hashtagged data.
import sys
import torch
import torch.nn.functional as F
from stringzilla import Str, File
from datasets import load_dataset
import os
import sqlite3
from progress.bar import Bar
from typing import Optional

def dprint(str):
    if os.getenv('DBG'):
        print(str)

class WordDict:
    def __init__(self, max_words):
        self.word_2_idx = {}
        self.idx_2_word = []
        self.max_words = max_words
        self.num_words = 0

    def get(self, word):
        #dprint("get {}".format(str(word)))
        if word in self.word_2_idx:
            return self.word_2_idx[word]
        if self.num_words == self.max_words:
            return None
        #dprint("fill miss {} -> {}".format(str(word), self.num_words))
        self.word_2_idx[word] = self.num_words
        self.idx_2_word.append(word)
        self.num_words += 1
        assert self.has(word)
        return self.num_words - 1

    def has(self, word):
        return word in self.word_2_idx

    def encode(self, words):
        return torch.Tensor([self.get(word) for word in words], dtype=torch.long)        
    
    def decode(self, indices):
        return [self.idx_2_word[idx] for idx in indices]

class MarginLoss(torch.nn.Module):
    def __init__(self,
                 margin=1e-1):
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
    def __init__(self,
                 emb_size=256,
                 words: Optional[WordDict]=None,
                 tags: Optional[WordDict]=None,
                 vocab_size=int(1e6),
                 tag_vocab_size=int(1e5),
                 margin=1e-1):
        torch.nn.Module.__init__(self)
        self.emb_size = emb_size 

        words = words or WordDict(vocab_size)
        tags = tags or WordDict(tag_vocab_size)
        self.words = words
        self.tags = tags

        self.word_embs = torch.nn.Embedding(vocab_size, emb_size)
        self.tag_embs = torch.nn.Embedding(tag_vocab_size, emb_size)
        self.margin = MarginLoss(margin)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _project(self, wd, tokens):
        try:
            tagless = list(filter(lambda s: len(s) > 0 and s[0] != '#', tokens))
            tokenized = [ wd.get(word) for word in tagless ]
            f = [ f for f in filter(lambda x: x is not None, tokenized) ]
            return torch.LongTensor(f).to(self.device)
        except:
            import pdb; pdb.set_trace()

    def project_example(self, words, tags):
        idx = self._project(self.words, words)
        targets_pos = self._project(self.tags, tags)
        targets_neg = torch.randint(0, self.tags.num_words, size=(32,)).to(self.device)
        return (idx, targets_pos, targets_neg)

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
    
    def load_dictionaries(self, embs):
        self.words = embs['words']
        self.tags = embs['tags']
 
    def embed_words(self, words):
        wordvecs = self.word_embs.forward(self._project(self.words, words))
        return torch.einsum('ij->j', wordvecs)
    
    def search_tags(self, vector):
        sims = F.cosine_similarity(vector, self.tag_embs.weight)
        return sims

def main(argv):
    model = TagSpaceModel()

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
            xs, ys, negs = model.project_example(words, hashtags)

            optim.zero_grad()
            loss = model.forward(xs, ys, negs)
            example += 1
            if (example % 100) == 1:
                print("loss {} words {} tags {}".format(loss, model.words.num_words, model.tags.num_words))
            loss.backward()
            optim.step()
    torch.save({
        'model_state': model.state_dict(),
        'dictionaries': {
            'words' : model.words,
            'tags'  : model.tags,
        },
    }, 'model.pt')

if __name__ == '__main__':
    main(sys.argv)
