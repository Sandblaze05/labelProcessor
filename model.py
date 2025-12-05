import json
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from main import tokenize


def get_features(sentence, index):
    token = sentence[index]
    
    features = {
        'bias': 1.0,
        'word.lower': token.lower(),
        'word.isupper': token.isupper(),
        'word.isdigit': token.isdigit(),
        'word.is_bracketed': (token.startswith('[') and token.endswith(']')),
        'word.length': len(token),
        'word[-3:]': token[-3:], 
        'word[:2]': token[:2],
    }
    
    if index > 0:
        prev_token = sentence[index - 1]
        features.update({
            '-1:word.lower': prev_token.lower(),
            '-1:word.isupper': prev_token.isupper(),
            '-1:word.is_bracketed': (prev_token.startswith('[') and prev_token.endswith(']')),
        })
    else:
        features['BOS'] = True # Beginning of Sentence

    # Context: Next Word
    if index < len(sentence) - 1:
        next_token = sentence[index + 1]
        features.update({
            '+1:word.lower': next_token.lower(),
            '+1:word.isupper': next_token.isupper(),
        })
    else:
        features['EOS'] = True # End of Sentence

    return features


def prepare_data(data):
    X = []
    y = []
    for entry in data:
        tokens = entry['tokens']
        labels = entry['training_data']
        
        sent_features = [get_features(tokens, i) for i in range(len(tokens))]
        
        X.append(sent_features)
        y.append(labels)
    return X, y


def train():
    print("Loading data...")
    with open('dump.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} examples.")
    
    # Convert to Features (X) and Labels (y)
    X, y = prepare_data(raw_data)
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize CRF Model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    print("Training model...")
    crf.fit(X_train, y_train)
    
    # Evaluation
    labels = list(crf.classes_)
    if 'O' in labels: labels.remove('O') 
    
    y_pred = crf.predict(X_test)
    print("\n--- Model Performance ---")
    print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))
    
    with open('anime_crf.pkl', 'wb') as f:
        pickle.dump(crf, f)
    print("Model saved as 'anime_crf.pkl'")

def test():
    with open('anime_crf.pkl', 'rb') as f:
        crf = pickle.load(f)

        def sent2features(sent):
            return [get_features(sent, i) for i in range(len(sent))]

        new_title_tokens = tokenize('[Erai-raws] Mushoku no Eiyuu: Betsu ni Skill Nanka Iranakattan da ga - 01 [1080p ADN WEBRip HEVC AAC][MultiSub][B41E05F9]')
        
        features = sent2features(new_title_tokens)
        prediction = crf.predict_single(features)

        print(list(zip(new_title_tokens, prediction)))

if __name__ == "__main__":
    # train()
    test()