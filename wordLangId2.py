# Sarah Greenwood

import string
import collections
import operator
import numpy as np

def read_train(path):
    """This funtion reads in the training data

    Args:
        path: path to the training corpus

    Returns:
        Returns a list of sentences from the training corpus.

    """
    filepath = open(path, 'rb')
    text = filepath.read().decode('utf8', 'surrogateescape')
    text = text.splitlines()
    filepath.close()
    return text

def read_test(path):
    """This funtion reads in the test data

    Args:
        path: path to the test corpus

    Returns:
        Returns a list of reformatted sentences from the test set.

    """
    filepath = open(path, 'rb')
    text = filepath.read().decode('utf8', 'surrogateescape')
    text = text.splitlines()
    text = reformat_text(text, False)
    filepath.close()
    return text


def reformat_text(text, join):
    """This function takes a list of sentences and reformats them to remove
    the casing and the punctuation. It also adds start of sentence characters
    and end of sentence characters.

    Args:
        text: list of sentences from a corpus
        join: bool indicating whether or not to return a list or a string

    Returns:
        Reformatted data, either formatted as a list of sentences or one string,
        depending on whether or not join is specified.

    """
    result = []
    for line in text:
        translator = str.maketrans('', '', string.punctuation)
        line = line.translate(translator)
        line = line.lower()
        # $ is start, & is end
        result.append('$ ' + line.rstrip() + ' & ')
    if join:
        result = ''.join(result)
    return result


def make_bigrams(text):
    """This function takes a sentence and returns all of the bigrams

    Args:
        text: a string/sentence

    Returns:
        Returns a list of reformatted sentences from the test set.

    """
    letter_bigrams = [text[i:i+2] for i in range(len(text)-1)]
    result = []
    for pair in letter_bigrams:
        combo = ' '.join(pair)
        result.append(combo)
    return result


def prep_train(path, threshold):
    """This function takes a filepath to a training corpus and a threshold
    for word frequencies and returns all training bigrams and all words in the corpus.
    It replaces any rare words (below threshold) with <UNK> to represent unknown.

    Args:
        path: path to training corpus
        threshold: int value for word frequency threshold

    Returns:
        Returns all training bigrams from test set and a list of all words in the
        training corpus.
    """
    train = read_train(path)
    train = reformat_text(train, True)
    words = train.split()
    words_freq = collections.Counter(words)
    low_freq = [k for k, v in words_freq.items() if v < threshold]
    words = ["<UNK>" if x in low_freq else x for x in words]
    train_bigrams = make_bigrams(words)
    bi_freq = normalizer(train_bigrams, 1, len(set(words))**2 - len(set(train_bigrams)))
    uni_freq = normalizer(words, threshold, 1)
    return(train_bigrams, words, bi_freq, uni_freq)


def good_turing(test, words, train_bigrams, bi_freq, uni_freq):
    """This function returns the probability of a sentence belonging to a given corpus
    after performing Good Turing smoothing.

    Args:
        test: test string/sentence
        words: list of all words in a training corpus
        training_bigrams: list of all bigrams in training corpus
        bi_freq: frequency distribution of bigrams in training corpus
        uni_freq: frequency distribution of unigrams in training corpus

    Returns:
        Returns the probability of a sentence belonging to a given corpus.
    """
    test = make_bigrams(test.split())
    result = 0
    for pair in test:
        split = pair.split()
        if pair in train_bigrams:
            bi = train_bigrams.count(pair) + 1
            uni = words.count(split[0]) + 1
        elif split[0] in words:
            bi = train_bigrams.count(split[0] + " " + "<UNK>") + 1
            uni = words.count(split[0]) + 1
        elif split[1] in words:
            bi = train_bigrams.count("<UNK>" + " " + split[1]) + 1
            uni = words.count("<UNK>") + 1
        else:
            bi = train_bigrams.count("<UNK> <UNK>") + 1
            uni = words.count("<UNK>") + 1
        if (bi in bi_freq.keys()) & (bi-1 in bi_freq.keys()):
            bi = bi * (bi_freq[bi]/bi_freq[bi-1])
        if (uni in uni_freq.keys()) & (uni-1 in bi_freq.keys()):
            uni = uni * (uni_freq[uni]/uni_freq[uni-1])
        p_bi = bi/len(train_bigrams)
        p_uni = uni/len(words)
        result += np.log(p_bi/p_uni)
    return result


def return_language(text, english_bigrams, english_words, eng_bi_freq, eng_uni_freq,
                    french_bigrams, french_words, fr_bi_freq, fr_uni_freq,
                    italian_bigrams, italian_words, it_bi_freq, it_uni_freq):
    """This function computes the probability of a test sentence
    existing in a given corpus (English, Italian, French) and returns
    which languages is most probable.

        Args:
            text: string representing a test sentence
            english_bigrams: all bigrams in the english corpus
            english_words: all words in the english corpus
            english_bi_freq: frequency distribution of english bigrams
            eng_uni_freq: frequency distribution of english unigrams
            french_bigrams: all bigrams in the french corpus
            french_words: all words in the french corpus
            fr_bi_freq: frequency distribution of french bigrams
            fr_uni_freq: frequency distribution of french unigrams
            italian_bigrams: all bigrams in the italian corpus
            italian_words: all words in the italian corpus
            it_bi_freq: frequency distribution of italian bigrams
            it_uni_freq: frequency distribution of italian unigrams

        Returns:
            A string (English, French, or Italian) representing
            which language the sentence is predicted to belong to.
    """
    result = {
        "English" : good_turing(text, english_words, english_bigrams, eng_bi_freq, eng_uni_freq),
        "Italian" : good_turing(text, italian_words, italian_bigrams, it_bi_freq, it_uni_freq),
        "French" : good_turing(text, french_words, french_bigrams, fr_bi_freq, fr_uni_freq)
    }
    result = max(result.items(), key=operator.itemgetter(1))[0]
    return result


def normalizer(grams, threshold, unknowns):
    """This function takes a corpus, a threshold for unknowns, and a
    zero frequency distribution value estimate and returns a dictionary
    of a frequency distribution to be used for Good Turing smoothing.

    Args:
        grams: string, corpus
        threshold: int value for word frequency threshold
        unknowns: int for unknown value in distribution

    Returns:
        Returns continuous portion of a frequency distribution for a given corpus. 
    """
    freq = collections.Counter(grams)
    result = {0: unknowns}
    while True:
        match = [k for k, v in freq.items() if v == threshold]
        if len(match) == 0:
            break
        result[threshold] = len(match)
        threshold += 1
    return result


def write_result(test, english_bigrams, english_words, eng_bi_freq, eng_uni_freq,
                 french_bigrams, french_words, fr_bi_freq, fr_uni_freq,
                 italian_bigrams, italian_words, it_bi_freq, it_uni_freq):
    """This function takes a test set (list of sentences) and outputs a
    file with the predicted languages for each sentence.

        Args:
            test: list of strings/sentences
            english_bigrams: all bigrams in the english corpus
            english_words: all words in the english corpus
            english_bi_freq: frequency distribution of english bigrams
            eng_uni_freq: frequency distribution of english unigrams
            french_bigrams: all bigrams in the french corpus
            french_words: all words in the french corpus
            fr_bi_freq: frequency distribution of french bigrams
            fr_uni_freq: frequency distribution of french unigrams
            italian_bigrams: all bigrams in the italian corpus
            italian_words: all words in the italian corpus
            it_bi_freq: frequency distribution of italian bigrams
            it_uni_freq: frequency distribution of italian unigrams

        Returns:
            A list of predicted languages to be used in the evaluate
            function.
    """
    counter = 0
    result = []
    with open('wordLangId2.out', 'w') as f:
        for line in test:
            counter = counter + 1
            language = return_language(line, english_bigrams, english_words, eng_bi_freq, eng_uni_freq,
                                       french_bigrams, french_words, fr_bi_freq, fr_uni_freq,
                                       italian_bigrams, italian_words, it_bi_freq, it_uni_freq)
            answer = str(counter) + " " + language
            result.append(answer)
            f.write(answer + "\n")
    return result

def evaluate(results):
    """This function takes a list of predicted languages returned from the
    write_result function and compares it to the real language solution.

        Args:
            result: a list of predicted languages

        Returns:
            A float representing the percentage of languages the model
            predicted correctly.
    """
    filepath = open('test/LangId.sol', 'r')
    solution = filepath.read().splitlines()
    return sum(1 for r, s in zip(results, solution) if r == s) / float(len(solution))


def main():
    """This function runs the program"""
    # prep english corpus
    english_bigrams, english_words, eng_bi_freq, eng_uni_freq = prep_train('train/LangId.train.English', 1)
    # prep french corpus
    french_bigrams, french_words, fr_bi_freq, fr_uni_freq = prep_train('train/LangId.train.French', 1)
    # prep italian corpus
    italian_bigrams, italian_words, it_bi_freq, it_uni_freq = prep_train('train/LangId.train.Italian', 1)
    # read in test set
    test = read_test('test/LangId.test')
    # predict languages and evaluate
    answer = write_result(test, english_bigrams, english_words, eng_bi_freq, eng_uni_freq,
                          french_bigrams, french_words, fr_bi_freq, fr_uni_freq,  
                          italian_bigrams, italian_words, it_bi_freq, it_uni_freq)
    # print evaluation
    print(evaluate(answer))


if __name__ == '__main__':
    main()
