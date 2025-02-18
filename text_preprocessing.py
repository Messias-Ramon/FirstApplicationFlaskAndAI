import re
import nltk
import string
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from nltk.stem.porter import PorterStemmer

def perform_text_preprocessing_step(text, step):
    if step == 'lowercasing':
        return text.lower()
    
    elif step == 'punctuation':
        return text.translate(str.maketrans('', '', string.punctuation))

    elif step == 'special character':
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        text = re.sub('\s+', ' ', text)
        return text
    
    elif step == 'url':
        return re.sub(r'\b(?:https?|ftp):\/\/(?:www\.)?[\w-]+(\.[a-z]{2,})+(?:\/[^\s]*)?\b' + \
                  r'|\b(?:www\.)?[\w-]+(\.[a-z]{2,})+(?:\/[^\s]*)?\b', '', text)
        
    elif step == 'html':
        return re.sub(r'<\/?\w+>', '', text)
    
    elif step == 'mispelling':
        spell = SpellChecker()
        corrected_text = []
        misspelled_text = spell.unknown(text.split())
        words = text.split()
        for word in words:
            if word in misspelled_text:
                correction = spell.correction(word)
                if correction is not None:
                    corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)
        return " ".join(corrected_text)
    
    elif step == 'stopword':
        stopwords_list = stopwords.words('english')
        return " ".join([word for word in text.split() if word not in stopwords_list])
    
    elif step == 'stemming':
        ps = PorterStemmer()
        return " ".join([ps.stem(word) for word in text.split()])
    
    elif step == 'lemmatization':
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "R": wordnet.ADV}

        # find POS tags
        pos_text = pos_tag(text.split(), tagset='universal')
        f_lemmatize = lambda word, pos: lemmatizer.lemmatize(word, wordnet_map[pos]) if pos in wordnet_map else word
        return " ".join([f_lemmatize(word,pos) for word,pos in pos_text])
    
    else:
        print('Invalid step name')
        return text
    
def _mask_alphanumeric(text: str)->str:
    return re.sub(r'\b[a-zA-Z]+[0-9]+[a-zA-Z0-9]*\b|\b[0-9]+[a-zA-Z]+[a-zA-Z0-9]*\b', 'ALPHANUMERICTAG', text)

def _mask_number(text: str)->str:
    return re.sub(r'\b[0-9]+\b', 'NUMBERTAG', text)
        
def preprocess_text(text, steps = ['lowercasing', 'punctuation', 'special character', 
                                   'url', 'html', 'stopword', 'stemming', 
                                   'lemmatization'],
                    skip = None, correct_spelling = False,
                    mask_alphanumeric=True, mask_number=True):
    if correct_spelling:
        steps.insert(5, 'mispelling')
    if skip is not None:
        steps = [step for step in steps if (step not in skip)]
    for step in steps:
        text = perform_text_preprocessing_step(text, step)
    if mask_alphanumeric:
        text = _mask_alphanumeric(text)
    if mask_number:
        text = _mask_number(text)
    return text