# Train a simple model on a collection of hashtagged data.
import sys
import stringzilla as sz


def main(argv):
    word_2_idx = {}
    text = sz.Str('the rebel alliance decided not to attack the planet')
    split = text.split()
    for word in split[1:]:
        import pdb; pdb.set_trace()
        print(word)

    if False:
        for line in text.splitlines():
            split = line.split()
            for word in split[1:]:
            #for word in split:
                import pdb; pdb.set_trace()
                print(word)

main(sys.argv)