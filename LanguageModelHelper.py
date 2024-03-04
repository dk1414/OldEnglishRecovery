import numpy as np
import random

class LanguageModelHelper:
    def __init__(self, vocab=None, word_frequency_map=None, file_path=None, min_freq=5, unk_token = '[UNK]'):
        """
        Initialize the LanguageModel with a given text file.

        Parameters:
        vocab (set): Vocabulary of words for language model.
        word_to_id (dict): Mapping of vocab words to ids.
        file_path (str): Path to the text file. Only include if you are not using a precomputed vocabulary/word_id map.
        min_freq (int): Minimum number of times a word must show up in data to be included in vocabulary.
        """

        self.min_freq = min_freq
        self.unk = unk_token
        self.file_path = file_path

        if file_path != None:
            v, f = self._generate_vocabulary(file_path)
            self.vocabulary = v
            self.word_frequency_map = f
        else:
            if vocab and word_frequency_map:
                self.vocabulary = vocab
            else:
                raise Exception("vocab and word_frequency_map cannot be empty if no text file is provided")

    def _generate_vocabulary(self, file_path):
        """
        Generate the vocabulary based on the content of the text file.

        Parameters:
        file_path (str): Path to the text file.

        Returns:
        set: The generated vocabulary.
        """
        text = []

        #read text to list. text will end up being a list of lists of words
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                    # Process each line as a sentence
                    words = (line.strip().split())
                    text.append(words)
    


        UNK_symbol = self.unk #unknown token
        vocab = set([UNK_symbol])

        # create term frequency of the words
        words_term_frequency_train = {}
        for doc in text:
            for word in doc:
                # this will calculate term frequency
                # since we are taking all words now
                words_term_frequency_train[word] = words_term_frequency_train.get(word,0) + 1

        # create vocabulary
        for doc in text:
            for word in doc:
                if words_term_frequency_train.get(word,0) >= self.min_freq: #only include frequent enough words
                    vocab.add(word)
        
        return vocab, words_term_frequency_train

    def find_candidates(self, word, missing_token="¿"):

        """
        Given a word with missing characters, return all possible candidate words in vocabulary.

        Parameters:
        word (str): Word with missing characters.

        Returns:
        list: List of possible word candidates.
        """
        candidates = []

        for vocab_word in self.vocabulary:
            if len(word) != len(vocab_word):
                continue  # Skip words with different lengths

            candidate = []
            for char1, char2 in zip(word, vocab_word):
                if char1 == missing_token:
                    candidate.append(char2)
                elif char1 == char2:
                    candidate.append(char2)  
                else:
                    break #mismatch, skip
            else:
                try:
                    candidates.append("".join(candidate))
                except:
                    print(word, vocab_word, candidate,"".join(candidate) )
                    raise Exception

        return candidates


    def get_dataset(self):
        '''
        Returns list of lists of strings where each sublist represents a line of text.
        
        '''
        if not self.file_path:
            raise Exception("No file path provided when initializing Class instance")
        text = []

        #read text to list. text will end up being a list of lists of words
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                    # Process each line as a sentence
                    words = (line.strip().split())
                    
                    for i, word in enumerate(words):
                        if word not in self.vocabulary:
                            words[i] = self.unk
                    
                    if len(words):
                        text.append(words)

        return text
    

    def get_masked_dataset(self, percent_masks_per_line = 0.1, mask_token = '¿'):
        '''
        Returns list of lists of strings where each sublist represents a line of text.
        In each line of text, a percentage of characters will be masked.
        Also returns list of indices where mask tokens are located. ex. [(i,j,k),(a,b,c)]
        
        '''
        if not self.file_path:
            raise Exception("No file path provided when initializing Class instance")
        
        text = []
        masked_indices = []
        #read text to list. text will end up being a list of lists of words
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                    # Process each line as a sentence
                    words = (line.strip().split())
                    
                    for i, word in enumerate(words):
                        if word not in self.vocabulary:
                            words[i] = self.unk
                    
        # Replace words not in vocabulary with unknown token
                    for i, word in enumerate(words):
                        if word not in self.vocabulary:
                            words[i] = self.unk
                    
                    if len(words):
                        # Mask characters

                        total_chars = sum([len(word) for word in words])

                        chars_to_mask = int(percent_masks_per_line * total_chars)

                        char_indices = set(random.sample(range(0,total_chars), chars_to_mask))
                        
                        char_count = 0
                        for i,word in enumerate(words):
                            new_word = []
                            for j, char in enumerate(word):
                                if char_count in char_indices:
                                    new_word.append(mask_token)
                                    masked_indices.append((len(text), i, j))
                                else:
                                    new_word.append(char)
                                char_count += 1
                            words[i] = ''.join(new_word)

                            
                        text.append(words)

            return text, masked_indices
    

    def most_likely(self, prob_distribution):
        """
        Predicts the most likely word from the vocabulary given a probability distribution.

        Parameters:
        prob_distribution (list): A list representing the probability distribution over the vocabulary.
    

        Returns:
        id: The most likely word id from the distribution.
        """
        return np.argmax(prob_distribution)
    


    def most_likely_with_candidates(self, word, prob_distribution, word_to_id, missing_token):
        """
        Predicts the most likely word from the vocabulary given a probability distribution, uses the find
        candidates 

        Parameters:
        word (str): Word with missing characters that we are trying to predict.
        prob_distribution (list): A list representing the probability distribution over the vocabulary.
        word_to_id (dict): Dict mapping words in vocab to id.
        missing_token (str): Character being used a missing token in word.

        Returns:
        id: The most likely word id from the distribution.
        """
        
        candidates = set([word_to_id[i] for i in self.find_candidates(word, missing_token=missing_token)])


        # Sort indices based on probabilities, each indice corresponds to a word id
        sorted_indices = np.argsort(prob_distribution).tolist()[::-1]

        #iterate through ids by how probable they are, if one is a possible match return it
        for id in sorted_indices:
            if id in candidates:
                return id
        
        #this line should never run, if it does something is going wrong
        raise Exception("Prob_distribution does should contain a probabilty for every word in vocab")


