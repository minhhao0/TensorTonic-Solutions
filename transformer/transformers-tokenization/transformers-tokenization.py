import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        vocab=[self.pad_token,self.unk_token,self.bos_token,self.eos_token]
        unique_words=[]
        for text in texts:
            for token in text.lower().split():
                if token not in unique_words:
                    unique_words.append(token)
        unique_words=sorted(unique_words)
        vocab+=unique_words
        self.vocab_size+=len(vocab)
        for index in range(self.vocab_size):
            self.word_to_id[vocab[index]]=index
            self.id_to_word[index]=vocab[index]
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        text=text.lower()
        encode_arr=[]
        for token in text.split():
            if token not in self.word_to_id.keys():
                encode_arr.append(self.word_to_id[self.unk_token])
            else:
                encode_arr.append(self.word_to_id[token])
        return encode_arr
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        result=[]
        for i in ids:
            if i not in self.id_to_word.keys():
                result.append(self.unk_token)
            else:
                result.append(self.id_to_word[i])
        return ' '.join(result)
