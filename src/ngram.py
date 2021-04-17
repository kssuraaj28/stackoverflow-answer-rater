SENTENCE_BEGIN = '-BEGIN-'

def gen_corpus(body_list,output_txt):
    '''
    Generate a corpus from a list of html strings
    '''
    from src.body_util import clear_tags
    import re

    ofile = open(output_txt,'w')
    for body in body_list:
        line=clear_tags(body).lower()
        line=re.sub('[^0-9a-zA-Z]+', ' ', line)
        ofile.write(line)
        ofile.write('\n')
    ofile.close()
    print (f"Generated corpus file: {output_txt}")

def make_language_model(corpus_txt):
    '''
    Generate a language model from a given corpus
    '''
    VOCAB_SIZE = 600000
    LONG_WORD_THRESHOLD = 5
    LENGTH_DISCOUNT = 0.15

    import collections
    unigramCounts = collections.Counter()
    bigramCounts = collections.Counter()
    bitotalCounts = collections.Counter()
    totalCounts = 0

    def sliding(xs, windowSize):
        for i in range(1, len(xs) + 1):
            yield xs[max(0, i - windowSize):i]

    def bigramWindow(win):
        '''
        Generates a tuple from an len 1 or 2 object
        '''
        assert len(win) in [1, 2]
        if len(win) == 1:
            return (SENTENCE_BEGIN, win[0])
        else:
            return tuple(win)

    with open(corpus_txt, 'r') as f:
        for l in f:
            ws = l.split()
            unigrams = [x[0] for x in sliding(ws, 1)]
            bigrams = [bigramWindow(x) for x in sliding(ws, 2)]
            totalCounts += len(unigrams)
            unigramCounts.update(unigrams) #I'm sure we can refactor this
            bigramCounts.update(bigrams)
            bitotalCounts.update([x[0] for x in bigrams])

    from math import log
    def unigramCost(x):

        if x not in unigramCounts:
            length = max(LONG_WORD_THRESHOLD, len(x))
            return -(length * log(LENGTH_DISCOUNT) + log(1.0) - log(VOCAB_SIZE))
        else:
            return log(totalCounts) - log(unigramCounts[x])

    def bigramModel(a, b):
        return log(bitotalCounts[a] + VOCAB_SIZE) - log(bigramCounts[(a, b)] + 1)

    return unigramCost, bigramModel