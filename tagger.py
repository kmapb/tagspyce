import sys
from tagspyce import TagSpaceModel, WordDict
import torch

def split_and_normalize(s):
    return s.lower().split()

def main(argv):
    model = TagSpaceModel()
    checkpoint = torch.load(argv[0])
    model.load_state_dict(checkpoint['model_state'])
    model.load_dictionaries(checkpoint['dictionaries'])
    while True:
        s = split_and_normalize(input('** '))
        repr = model.embed_words(s)
        sims = model.search_tags(repr)
        k = 5
        top_sims, indices = torch.topk(sims, k)
        for i in range(k):
            print("{}: {}".format(model.tags.decode([indices[i]]), sims[i]))

if __name__ == '__main__':
    main(sys.argv[1:])
