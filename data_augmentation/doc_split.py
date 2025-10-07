from typing import List
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from operator import ge
import spacy
from tqdm import tqdm

# from .abstract_passage_chunker import AbstractPassageChunker
from typing import Dict, List

## Sentence based segmentation
def sentence_chunker(text: str, sentences_per_chunk: int) -> List[str]:
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    chunks = []
    
    # Group sentences into chunks
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i: i + sentences_per_chunk]
        chunk = ' '.join(chunk_sentences)
        chunks.append(chunk)
    
    return chunks


## Semantic similarity based segmentation
def semantic_chunker(text: str, similarity_threshold: float = 0.85) -> List[str]:
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Load pre-trained embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode sentences into embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    # Compare sentence similarity to decide chunk boundaries
    for i in range(1, len(sentences)):
        similarity = util.pytorch_cos_sim(embeddings[i - 1], embeddings[i]).item()
        if similarity > similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

## Fixed size segmentation
def word_splitter(source_text: str) -> List[str]:
    # Replace multiple whitespaces with a single space
    source_text = re.sub(r'\s+', ' ', source_text)
    # Split text by single whitespace
    return re.split(r'\s', source_text)

def get_chunks_fixed_size(text: str, chunk_size: int) -> List[str]:
    text_words = word_splitter(text)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk_words = text_words[i: i + chunk_size]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
    return chunks

def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float) -> List[str]:
    text_words = word_splitter(text)
    overlap_int = int(chunk_size * overlap_fraction)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        start_index = max(i - overlap_int, 0)
        end_index = i + chunk_size
        chunk_words = text_words[start_index: end_index]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
    return chunks


## Sliding window segmentation
def segment_document(doc_text,window_size=10,stride=5):    
    sentences = sent_tokenize(doc_text)
    
    
    passages = []
    # window_size = 10
    # stride = 5
    
    for i in range(0, len(sentences) - window_size + 1, stride):
        passage = " ".join(sentences[i:i + window_size])
        passages.append(passage)
    
    return passages






nlp = spacy.load("en_core_web_sm", exclude=[
                 "parser", "tagger", "ner", "attribute_ruler", "lemmatizer", "tok2vec"])
nlp.enable_pipe("senter")
nlp.max_length = 2000000  # for documents that are longer than the spacy character limit

def chunk_document(document_sentences, sentences_word_count, passage_size=250) -> List[Dict]:
    """
    Creates the passage chunks for a given document
    """
    passages = []

    current_passage = ''
    current_passage_word_count = 0
    sub_id = 1

    for sentence, word_count in zip(document_sentences, sentences_word_count):
        if word_count >= passage_size:
            if current_passage:
                passages.extend([current_passage,sentence.text])
                sub_id += 2

            else:
                passages.append(sentence.text)
                sub_id += 1

        elif word_count + current_passage_word_count > passage_size:
            passages.append(current_passage)
            current_passage = sentence.text
            current_passage_word_count = word_count
            sub_id += 1

        else:
            current_passage += ' ' + sentence.text + ' '
            current_passage_word_count += word_count

    if current_passage:
        passages.append(current_passage)

    return passages
