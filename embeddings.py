import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
# import torch
# from torch.utils.data import DataLoader, Dataset
import h5py
from functools import partial
import pickle
import nltk
# torch.set_num_threads(4)


def get_w2v_embeddings(text, wv_object, aggregated_sentence_embedding, allow_close_word = False):
    words = text.split()
    
    embeddings = []#np.zeros((sentence_size, wv_object.vector_size), dtype=np.float32)
    for word in words:
        if word in wv_object.index_to_key:
            embeddings.append(wv_object[word])
        elif allow_close_word == True:
            close_word = get_closest_word(word, wv_object.index_to_key)
            embeddings.append(wv_object[close_word])
    
    if aggregated_sentence_embedding:
        #embeddings = embeddings[list(set(np.where(embeddings!=0)[0].tolist()))]
        if len(embeddings)==0:
            return np.zeros((wv_object.vector_size,))
        return sum(embeddings)/len(embeddings)
    else:
        return embeddings

def get_closest_word(word:str, candidates:list)-> str:
    candidates_with_distances = [(candidate,nltk.edit_distance(candidate, word)) for candidate in candidates]
    closest_word = min(candidates_with_distances, key=lambda p:p[1])[0]
    return closest_word

# def get_bert_embeddings(texts, tokenizer, model):

#     tokens = tokenizer(
#         texts,
#         return_tensors="pt",           # Return PyTorch tensors
#         truncation=True,               
#         max_length=256,                # Limit 256 tokens
#         padding="max_length",         
#     )
    
   
#     with torch.no_grad():
#         outputs = model(**tokens)
    
#     sentence_embeddings = outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
#     return sentence_embeddings.cpu().numpy()

def generate_embeddings(texts: pd.Series, method: str, hdf5_file_path: str, text_embedder_output_filepath: str='text_embedder.pkl', wv_model_path: str = None, wv_object=None, max_features=300,
                        aggregated_sentence_embedding=True, batch_size=None, allow_close_word=False):
    """
    Returns a numpy array with the word embeddings of shape len(texts) by number_of_features.

    Supported methods:
      - 'tf-idf'
      - 'word2vec'
      - 'distilbert'

    Input:
      - texts (pandas.Series): a pandas Series where each entry is a sentence (text string)
      - method (str): the embedding method ('tf-idf', 'word2vec', or 'distilbert')
      - output_type (str): either 'np-array' or 'pd-series'

    Output:
      - numpy array of shape len(texts) x number_of_embedding_features containing embeddings
    """
    hdf = h5py.File(hdf5_file_path, "w")

    if method == 'tf-idf':
        tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
        embeddings = tfidf.fit_transform(texts).toarray()
        dataset = hdf.create_dataset("embeddings", data=embeddings, dtype='float32')
        embedder_model_object = tfidf

    elif method == 'word2vec':
        if wv_model_path is None:
            raise ValueError(f"The parameter 'wv_model_path' cannot be None when method is 'word2vec'")
        
        if wv_model_path.endswith('.wordvectors'):
            wv_object = KeyedVectors.load(wv_model_path, mmap='r')
        elif wv_model_path.endswith('.model'):
            wv_object = Word2Vec.load(wv_model_path).wv
        embedder_model_object = wv_object

        get_w2v_embeddings_partial = partial(get_w2v_embeddings, wv_object=wv_object,
                                              aggregated_sentence_embedding=aggregated_sentence_embedding, allow_close_word=allow_close_word
                                            )
        embeddings = np.array([get_w2v_embeddings_partial(text) for text in texts])
        dataset = hdf.create_dataset("embeddings", data=embeddings, dtype='float32') #compression='gzip', compression_opts=4, 

    elif method == 'distilbert':
        if batch_size is None:
            batch_size = 32
        bert_model_name='distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        model = AutoModel.from_pretrained(bert_model_name, from_tf=True)
        embedder_model_object = model
        
        texts = texts.fillna("").astype(str).to_list()


        dataset = hdf.create_dataset("embeddings", shape=(len(texts), 768), dtype='float32') #compression='gzip', compression_opts=4, 

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
 
            batch_embeddings = get_bert_embeddings(batch_texts, tokenizer, model)
            
            dataset[start:end] = batch_embeddings
            print(f"Processed {end}/{len(texts)} ")
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    text_embedder = TextEmbedder(embedding_method=method, 
                                 total_features=max_features, 
                                 embedder_model_object=embedder_model_object, 
                                 aggregated_sentence_embedding=aggregated_sentence_embedding)

    with open(text_embedder_output_filepath, 'wb') as f:
        pickle.dump(text_embedder, f)


    return hdf

def load_hdf5_embeddings(hdf5_file_path: str):
    hdf = h5py.File(hdf5_file_path, "r")
    return hdf



class TextEmbedder:
    def __init__(self, embedding_method, total_features, embedder_model_object, aggregated_sentence_embedding=True, allow_close_word=False, bert_tokenizer=None):
        self.embedding_method = embedding_method
        self.total_features = total_features
        self.embedder_model_object = embedder_model_object
        self.aggregated_sentence_embedding = aggregated_sentence_embedding
        self.allow_close_word = allow_close_word
        self.bert_tokenizer = bert_tokenizer

    def embed(self, text:str):

        if self.embedding_method == 'tf-idf':
            embeddings = self.embedder_model_object.transform([text]).toarray()

        elif self.embedding_method == 'word2vec':
            embeddings = get_w2v_embeddings(text, wv_object=self.embedder_model_object,
                                            aggregated_sentence_embedding=self.aggregated_sentence_embedding, allow_close_word=self.allow_close_word)

        elif self.embedding_method == 'distilbert':
            embeddings = get_bert_embeddings([text], self.bert_tokenizer, self.embedder_model_object)

        return embeddings 



