# Sarah Greenwood

import operator
import string
import numpy as np


def read_train(path):
    """This funtion reads in the training data

    Args:
        path: path to the training corpus

    Returns:
        Returns a string of the training corpus.

    """
    fp = open(path, 'rb')
    text = fp.read().decode('utf8', 'surrogateescape')
    fp.close()
    return text


def read_test(path):
    """This funtion reads in the test data

    Args:
        path: path to the test corpus

    Returns:
        Returns a list of reformatted sentences from the test set.

    """
    fp = open(path, 'rb')
    text = fp.read().decode('utf8', 'surrogateescape')
    text = reformat_text(text.splitlines(), False)
    fp.close()
    return text


def make_bigrams(text):
    """This function takes a sentence and returns all of the bigrams

    Args:
        text:

    Returns:
        Returns a list of reformatted sentences from the test set.

    """
    text = [text[i:i+2] for i in range(len(text)-1)]
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
        # $ is start
        result.append("$" + line)
    if join:
        result = ''.join(result)
    return result


def txt_to_unigrams(text, threshold=30):
    """This function takes a training corpus and replaces all letters that
    appear fewer than 30 times with an <UNK> token. Then the function returns
    a dictionary of unigrams and counts, as well as a list of unknown values
    that were removed from the training corpus.

    Args:
        text: a training corpus string
        threshold: value used to remove infrequent letters

    Returns:
        Returns a dictionary of unigrams and their counts, as well as a list of
        removed, unknown characters.

    """
    # leter unigrams
    letters = [text[i:i+1] for i in range(len(text)-1)]

    # takes all letters and any that exist < threshold are labeled as <UNK>
    unigrams = {'<UNK>': 0}
    unknowns = [] # keep track of unknowns
    for letter in list(set(letters)):
        count = letters.count(letter)
        if letters.count(letter) < threshold:
            unknowns.append(letter)
            unigrams['<UNK>'] = unigrams['<UNK>'] + count
        else:
            unigrams[letter] = count
    return(unigrams, unknowns)


def replace_unknowns(bigrams, unknowns):
    """This function takes a list of bigrams and a list of characters that should
    be considered to be <UNK> and replaces them.

    Args:
        bigrams: all bigrams in the training corpus
        unknowns: list of characters that should be considered unknown

    Returns:
        A list of bigrams with all unknown characters replaced with an <UNK> character.

    """
    for char in unknowns:
        bigrams = [i.replace(char, "<UNK>") for i in bigrams]
    return bigrams


def txt_to_bigrams(text, unknowns, unigrams):
    """This function takes a sentence and returns all of the bigrams.

    Args:
        text: a string representing a sentence
        unknowns: a list of unknowns
        unigrams: all unigrams

    Returns:
        A dictionary with all bigram counts.

    """
    # letter bigrams
    letter_bigrams = make_bigrams(text)
    letter_bigrams = replace_unknowns(letter_bigrams, unknowns)
    bigrams = {}
    for first in unigrams.keys():
        for second in unigrams.keys():
            bigrams[first + second] = letter_bigrams.count(first + second)
    return bigrams


def add_one_smoothing(unigrams, bigrams):
    """This function performs add one smoothing and returns the probatilities
    for all bigrams in a corpus.

    Args:
        unigrams: a dictionary of unigrams and their counts
        bigrams: a dictionary of all bigrams and their counts

    Returns:
        Returns conditional probabilties for all bigrams.
    """
    probabilities = {}
    V = len(unigrams.keys())
    for first in unigrams.keys():
        for second in unigrams.keys():
            numerator = bigrams[first+second] + 1
            denominator = unigrams[first] + V
            probabilities[first+second] = numerator / denominator
    return probabilities


def build_model(training_file):
    """This function builds a letter model from the beginning to end after
    being given a path to a training corpus.

    Args:
        training_file: path to training corpus

    Returns:
        Add one probability dictionary for bigrams and a list of all
        unigrams in the training corpus (vocab).
    """
    text = read_train(training_file)
    text = reformat_text(text.splitlines(), True)
    unigrams, unknowns = txt_to_unigrams(text)
    bigrams = txt_to_bigrams(text, unknowns, unigrams)
    add_one = add_one_smoothing(unigrams, bigrams)
    return(add_one, unigrams)


def prep_language(text, vocab):
    """This function prepares the test sentence by replacing unknown
    values with <UNK> tokens.

    Args:
        text: string representing a test sentence
        vocab: all words in training corpus

    Returns:
        string representing a test sentence with all unknown characters
        replaced with <UNK> tokens.
    """
    test_vocab = set(text)
    unknown = list(test_vocab - set(vocab))
    test_bigrams = make_bigrams(text)
    final = replace_unknowns(test_bigrams, list(unknown))
    return final


def calc_log_prob(text, vocab, model):
    """This function calculates the probability that a sentence is in
    a given corpus given the sentence, vocabulary, and model.

    Args:
        text: string representing a test sentence
        vocab: all words in a given corpus
        model: dictionary of bigrams and their smoothed probabilities

    Returns:
        Returns the probability of that training string existing in
        the given corpus.
    """
    result = 0
    text = prep_language(text, vocab)
    for val in text:
        result = result + np.log(model[val])
    return result


def return_language(text, english_vocab, english_model,
                    italian_vocab, italian_model,
                    french_vocab, french_model):
    """This function computes the probability of a test sentence
    existing in a given corpus (English, Italian, French) and returns
    which languages is most probable.

        Args:
            text: string representing a test sentence
            english_vocab: all characters in the english corpus
            english_model: dictionary of english bigrams and their probabilities
            italian_vocab: all characters in the italian corpus
            italian_model: dictionary of italian bigrams and their probabilities
            french_vocab: all characters in the french corpus
            french_model: dictionary of english french and their probabilities


        Returns:
            A string (English, French, or Italian) representing
            which language the sentence is predicted to belong to.
    """
    result = {
        "English" : calc_log_prob(text, english_vocab, english_model),
        "Italian" : calc_log_prob(text, italian_vocab, italian_model),
        "French" : calc_log_prob(text, french_vocab, french_model)
    }
    result = max(result.items(), key=operator.itemgetter(1))[0]
    return result


def write_result(test, english_bigrams, english_words,
                 french_bigrams, french_words,
                 italian_bigrams, italian_words):
    """This function takes a test set (list of sentences) and outputs a
    file with the predicted languages for each sentence.

        Args:
            test: list of strings/sentences
            english_bigrams: all bigrams in the english corpus
            english_words: all words in the english corpus
            french_bigrams: all bigrams in the french corpus
            french_words: all words in the french corpus
            italian_bigrams: all bigrams in the italian corpus
            italian_words: all words in the italian corpus

        Returns:
            A list of predicted languages to be used in the evaluate
            function.
    """
    counter = 0
    result = []
    with open('letterLangId.out', 'w') as f:
        for line in test:
            counter = counter + 1
            language = return_language(line, english_bigrams, english_words,
                                       french_bigrams, french_words,
                                       italian_bigrams, italian_words)
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
    fp = open('test/LangId.sol', 'r')
    solution = fp.read().splitlines()
    return sum(1 for r, s in zip(results, solution) if r == s) / float(len(solution))


def main():
    """This function runs the program"""
    # build english model
    english, english_vocab = build_model('train/LangId.train.English')
    # build french model
    french, french_vocab = build_model('train/LangId.train.French')
    # build italian model
    italian, italian_vocab = build_model('train/LangId.train.Italian')
    # read in and prep test set
    test = read_test('test/LangId.test')
    # output predicted languages
    answer = write_result(test, english_vocab, english,
                          italian_vocab, italian,
                          french_vocab, french)
    # print model accuracy eval
    print(evaluate(answer))


if __name__ == '__main__':
    main()
