import pandas as pd
import numpy as np
import pickle
from text_preprocessing import preprocess_text

#api with the purpose of predicting the possible subsystem description the client might need, based off their complains.

def predict_class(text: str):
    """This function tries to predict to which 'Subsystem Description' the text might be related to.
    Takes a string as a parameter, turns it into a pandas series, then applies the text preprocessing to it. 
    Right after, it turns every token in the sentence into a number — the embedding process, as a specialist might call it.
    Then, the model takes the embeddings and tries to predict it, based off the FMUCD public dataset, and returns a string containing the
    name of the class with higher certainty and another string containing its percentage."""

    text_series = pd.Series([text])
    
    text_series = text_series.apply(preprocess_text)

    with open('word2vec_embedder.pkl', 'rb') as f:
        w2v_embedder = pickle.load(f)
    
    text_embeddings = w2v_embedder.embed(text_series.values[0])

    with open('modelo_rfc.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction_proba = model.predict_proba([text_embeddings])[0]

    predicted_class_name = model.predict([text_embeddings])[0]

    predicted_class_index = np.where(model.classes_ == predicted_class_name)[0][0]

    certainty = prediction_proba[predicted_class_index]

    return predicted_class_name, f"{certainty*100}%"

def predict_multiple_classes(text: str):
    """This function tries to predict to which 'Subsystem Description' the text might be related to.
    Takes a string as a parameter, turns it into a pandas series, then applies the text preprocessing to it. 
    Right after, it turns every token in the sentence into a number — the embedding process, as a specialist might call it.
    Then, the model takes the embeddings and tries to predict it, based off the FMUCD public dataset, and returns a list containing
    tuples in which the first element is the probable class string and the second one is a string with the class certainty.
    So, in order to access such values, remember to use the python indexes."""
    text_series = pd.Series([text])
    text_series = text_series.apply(preprocess_text)

    with open('word2vec_embedder.pkl', 'rb') as f:
        w2v_embedder = pickle.load(f)

    text_embeddings = w2v_embedder.embed(text_series.values[0])

    with open('modelo_rfc.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction_proba = model.predict_proba([text_embeddings])[0]

    classes = model.classes_

    certainty_per_class = [(cls, f"{prob*100}%") for cls, prob in zip(classes, prediction_proba)]

    certainty_per_class_sorted = sorted(certainty_per_class, key=lambda x: -float(x[1].strip('%')))

    return certainty_per_class_sorted    