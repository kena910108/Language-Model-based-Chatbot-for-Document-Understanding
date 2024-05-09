
from transformers import AutoTokenizer

#from langchain import HuggingFacePipeline

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from rank_bm25 import BM25Okapi
import numpy as np

import fitz  # PyMuPDF

def get_wordnet_pos(treebank_tag):
    """
    Convert the part-of-speech naming scheme
    from the treebank to the wordnet scheme.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
        

def extract_text_by_layout(pdf_path):
    block_corpus = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text_blocks = page.get_text("blocks")  # Get text blocks
        previous_block_id = 0
        for block in text_blocks:  # Iterate through each block
            if block[6] == 0: # We only take the text
                if previous_block_id != block[5]:
                    x0, y0, x1, y1, text, block_type, line_number = block  # Unpack the tuple
                    block_corpus.append(text)
    return block_corpus

def get_context(pdf_path):
    og_corpus = extract_text_by_layout(pdf_path)
    sentence_corpus = [og_corpus[i-2] +
                       og_corpus[i-1] +
                       og_corpus[i] + 
                       og_corpus[i+1] + 
                       og_corpus[i+2]
                       for i in range(2, len(og_corpus)-2)]
    return sentence_corpus

def tokenize_corpus(tokenizer, lemmatizer, corpus):
    tokenized_sentence = [
                        [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in nltk.pos_tag(tokenizer.tokenize(sentence))] 
                        for sentence in corpus
                    ]
    
    return tokenized_sentence

def tokenize_query(tokenizer, lemmatizer, query):
    tokenized_sentence = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in nltk.pos_tag(tokenizer.tokenize(query))]
    
    return tokenized_sentence


def get_top_blocks(bm25, tokenized_query, sentence_corpus):
    doc_scores = bm25.get_scores(tokenized_query)

    sorted_docs = np.argsort(doc_scores)[::-1]

    # Retrieve the top N documents; let's say top 3
    top_docs = [sentence_corpus[idx] for idx in sorted_docs[:3]]

    return top_docs
