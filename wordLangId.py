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
    return(train_bigrams, words)


def add_one(test, words, train_bigrams):
    """This function returns the probability of a test sentence existing
    in a trainign corpus after performing add one smoothing

    Args:
        test: string representing a test sentence
        words: all words in training corpus
        train_bigrams: a list of all bigrams from training corpus

    Returns:
        Returns the probability of that training string existing in
        the given corpus.
    """
    test = make_bigrams(test.split())
    vocab_size = len(set(words))
    result = 0
    for pair in test:
        split = pair.split()
        if pair in train_bigrams:
            numerator = train_bigrams.count(pair) + 1
            denominator = words.count(split[0]) + vocab_size
            result = result + np.log(numerator/denominator)
        elif split[0] in words:
            # first word in vocab
            numerator = train_bigrams.count(split[0] + " " + "<UNK>") + 1
            denominator = words.count(split[0]) + vocab_size
            result = result + np.log(numerator/denominator)
        elif split[1] in words:
            # second word in vocab
            numerator = train_bigrams.count("<UNK>" + " " + split[1]) + 1
            denominator = words.count("<UNK>") + vocab_size
            result = result + np.log(numerator/denominator)
        else:
            numerator = train_bigrams.count("<UNK> <UNK>") + 1
            denominator = words.count("<UNK>") + vocab_size
            result = result + np.log(numerator/denominator)
    return result


def return_language(text, english_bigrams, english_words,
                    french_bigrams, french_words,
                    italian_bigrams, italian_words):
    """This function computes the probability of a test sentence
    existing in a given corpus (English, Italian, French) and returns
    which languages is most probable.

        Args:
            text: string representing a test sentence
            english_bigrams: all bigrams in the english corpus
            english_words: all words in the english corpus
            french_bigrams: all bigrams in the french corpus
            french_words: all words in the french corpus
            italian_bigrams: all bigrams in the italian corpus
            italian_words: all words in the italian corpus

        Returns:
            A string (English, French, or Italian) representing
            which language the sentence is predicted to belong to.
    """
    result = {
        "English" : add_one(text, english_words, english_bigrams),
        "Italian" : add_one(text, italian_words, italian_bigrams),
        "French" : add_one(text, french_words, french_bigrams)
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
    with open('wordLangId.out', 'w') as filepath:
        for line in test:
            counter = counter + 1
            language = return_language(line, english_bigrams, english_words,
                                       french_bigrams, french_words,
                                       italian_bigrams, italian_words)
            answer = str(counter) + " " + language
            result.append(answer)
            filepath.write(answer + "\n")
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
    english_bigrams, english_words = prep_train('train/LangId.train.English', 1)
    # prep french corpus
    french_bigrams, french_words = prep_train('train/LangId.train.French', 1)
    # prep italian corpus
    italian_bigrams, italian_words = prep_train('train/LangId.train.Italian', 1)
    # read in test set
    test = read_test('test/LangId.test')
    # write out answer
    answer = write_result(test, english_bigrams, english_words,
                          french_bigrams, french_words,
                          italian_bigrams, italian_words)
    # evaluate model
    print(evaluate(answer))


if __name__ == '__main__':
    main()
