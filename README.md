### Sarah Greenwood
Running instructions: 
1. Store training corpora in a folder labeled train, located on the same level as the model .py files. Have them labeled as they are in the assignment
2. Store the testing corpus in a folder labeled train, located on the same level as the model .py files. Have them labeled as they are in the assignment
3. Run `python filename.py`, wait and read reported accuracy. 

### Question 1: 
Can the letter bigram model be implemented without any kind of smoothing?
- The letter bigram model cannot be implemented without any smoothing because there are values with zero frequency in the test corpuses that will not exist in the training corpus. 

What do you decide to do and why did you do it that way? And, if you decide to do smoothing, what kind of smoothing do you need to use to avoid zero-counts in the data? Would add-one smoothing be appropriate or you need better algorithms? Why (not)?
My solution for these unknown characters was twofold. 
- First, I chose a threshold of 30 for labeling unknown characters. Meaning that if a character appeared fewer than 30 times in the training corpus, it was labeled as unknown. This way, when unknown characters were in the test corpus, they could be labeled as such and would exist in bigram and unigram probability tables. Also, this was an okay thing to do with the etter model because letters appear frequently (due to how the alphabet, even in differen languages, is used). So removing infrequent letters from this corpus was more of a cleaning exercise than shrinking the corpus. 
- Second, I chose to use add one smoothing so that we did not have probabilities with a value of zero (which would harm our conditional probability calculations).

Compare your output file with the solution file (LangId.sol). How many times was your program correct?
- My program has 99% accuracy



### Question 2: 
Can the word bigram model be implemented without any kind of smoothing? 
- No, again, we need smoothing to handle words that are in the test corpus that do not appear in the training corpus. They cannot have zero probability, so we need smoothing.

What do you decide to do and why did you do it that way? And, if you decide to do smoothing, try add-one smoothing.
Again, my solution here was twofold. 
- First chose to have a threshold of zero for handling unknown characters. This was a deliberate decision because our corpus was so small, and having a threshold of 1 would label too much of our training corpus as <UNK>.
- Second, as suggested, I performed add one smoothing to handle the unseen bigrams and unigrams in the test corpus. This effectively shifted some probability to the zero frequency counts and allowed us to compute the probabilities for each sentence, for each language corpus. 

Compare your output file with the solution file provided in the Assignment2 folder (LangId.sol). How many times was your program correct?
- My program had 99.66% accuracy


### Question 3: 
- I performed Good Turing smoothing with a zero threshold for cutoffs. My cutoff choice had the same motivation as it did for question 2. 
- For the zero-frequency adjustments for Good Turing, for unigrams I used the value of 1 (meaning that only 1 value is not seen in the training corpus), which follows assumptions used to derive Good Turing. For bigram zero frequency adjustments, for unseen bigrams, I used #(unigrams-squared) - #(seen bigrams)

Compare your output file with the solution file provided in the Assignment2 folder (LangId.sol). How many times was your program correct?
- My program had 100% accuracy

Which of the language models at Question#1, Question#2, and Question#3 is the best? 
- Each model has its own advantages and disadvantages. Looking at accuracy, alone, the model in Question 3 performs the best. With these datasets, Good Turing effectively shifted probabilities to evaluate the language in the test sentences.

Comment on advantages and disadvantages of these language models on the task (be as detailed as possible based on your observations).
- While the Good Turing model (Question 3), objectively, performed the best. It was the most complicated to implement and required keeping track of an entire word-based corpus. 
- Good Turing also has issues with evaluating the frequency distribution when the distribution itself gets spotty. This requires generalizations. 
- If we wanted to look for an effective, simple model that performs well on the question at hand (language identification), the Question 1 letter based model could be considered the best. This is because keeping track of an alphabet, rather than a corpus, is much simpler. This, in turn, has a much shorter running time. 