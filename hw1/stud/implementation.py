import numpy as np
from typing import List, Tuple
from model import Model
import json
import torch


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    vocab, labels = read_data()    
    stud_model = StudentModel(vocabulary= vocab, labels= labels, device= device)
    return stud_model

def read_data():
    vocab_data = "model/vocab_data.json"
    targets_data =  "model/label_data.json"

    with open(vocab_data, 'r') as voc:
        vocabs = json.load(voc)

    with open(targets_data, 'r') as lab:
        labels = json.load(lab)

    return vocabs, labels

def encode(data, vocab, device):

    length_sentences = []
    encoded_sentences = []
    for sentence in data:
        encoded_words = []
        for word in sentence:
            try:
                element = vocab[word]
            except:
                element = vocab["<UNK>"]
            encoded_words.append(element)
        length_sentences.append(len(sentence)) # saving original sentence lengths        
        while len(encoded_words) < 200:
            encoded_words.append(vocab["<PAD>"])
        
        encoded_sentences.append(torch.LongTensor(encoded_words))
    return torch.stack(encoded_sentences), length_sentences

def decode(data, labels, length):
    # to be able to erase padding predictions,
    # sentence lengths are saved in encode function
    print("decode")
    decoded_sequences = []
    for seq_index in range(len(data)):
        decoded_labels = []
        for label_index in range(length[seq_index]):
            sequence = data[seq_index]
            label_id = sequence[label_index]
            for key in labels.keys():
                if label_id == labels[key]:
                    decoded_labels.append(key)
        decoded_sequences.append(decoded_labels)
    
    return decoded_sequences

class RandomBaseline(Model):

    options = [
        ('LOC', 98412),
        ('O', 2512990),
        ('ORG', 71633),
        ('PER', 115758)
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]


class StudentModel(Model):
    def __init__(self, vocabulary, labels, device):
        # STUDENT: construct here your model
        # this class should be loading your weights and vocabulary
        Model.__init__(self, vocabulary, labels)
        self.vocabulary = vocabulary
        self.labels = labels
        self.device = device

        model_path = "model/epoch_24_f1_0.9232"
        self.load_state_dict(torch.load(model_path, map_location = self.device))
        

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        encoded_data, data_lengths = encode(tokens, self.vocabulary, self.device)
        print("rpiden")
        logits = self(encoded_data)
        predictions = torch.argmax(logits, -1)
        results = decode(predictions, self.labels, data_lengths)
        return results

        # remember to respect the same order of tokens!