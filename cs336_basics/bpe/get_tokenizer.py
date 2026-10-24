import pickle
from cs336_basics.bpe import tokenization2 as bpe
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, 
                 special_tokens: list[str] | None = None) :
        self.vocab =vocab
        self.merges = merges
        
        self.special_tokens : list[str] = special_tokens if special_tokens is not None else []
        
        # revocab tokens->id
        self.revocab = {
            token : token_id
            for token_id,token in vocab.items()
        }
        
        for sp_token in self.special_tokens:
            sp_bytes = sp_token.encode("utf-8")
            if sp_token not in self.revocab:
                new_id = len(vocab)
                self.vocab[new_id] = sp_bytes
                self.revocab[sp_bytes] = new_id 
            

    @classmethod      
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath,"rb") as f :
            vocab = pickle.load(f)
            
        with open(merges_filepath,"rb") as f :
            merges = pickle.load(f)
        
        return cls(vocab,merges,special_tokens)
    
    
    def encode(self, text: str) -> list[int]:
        
        token_ids : list[int] = []
        
        # text_spilted
        ts = bpe.split_by_special(text,self.special_tokens,drop_special = False)
        
        for chunk in ts :
            if not chunk: 
                continue
            
            
            if chunk in self.special_tokens:
                token_id = self.revocab[chunk.encode("utf-8")]
                token_ids.append(token_id)
                continue
            
            for word in bpe.PAT.finditer(chunk):
                
                word = bpe.word2bytes(word.group(0))
                
                for merge in self.merges:
                    
                    word =  bpe.apply_merge(word,merge)
                    
                token_ids.extend(self.revocab[token] for token in word)
                
        return token_ids
            
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable :
            token_ids = self.encode(text)
            
            for token_id in token_ids:
                yield token_id
            
        
        
    def decode(self, ids: list[int]) -> str:
        
        decode_text = b""
        decode_text = decode_text.join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")   
        return decode_text  
        
        