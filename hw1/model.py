from typing import List
import torch
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self, vocabulary, labels):
        super(Model, self).__init__()
        self.vocabulary = vocabulary
        self.labels = labels
        self.word_embedding = nn.Embedding(len(self.vocabulary), 
                                            100, # embedding dimension
                                            padding_idx= self.vocabulary["<PAD>"])
        self.lstm = nn.LSTM(100, # embedding dimension
                            100, # hidden dimension
                            bidirectional= True,
                            num_layers= 2, # number of LSTM layer
                            dropout= 0.5,
                            batch_first= True)

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(200, # since it's bidirectional, hidden dimension * 2 => LSTM out dim
                                    len(labels))

    def forward(self, data):
        embedding_data = self.word_embedding(data)
        embedding_data = self.dropout(embedding_data)
        output, (h, c) = self.lstm(embedding_data)
        output = self.dropout(output)
        output = self.classifier(output)
        return output

        def predict(self, tokens: List[List[str]]) -> List[List[str]]:
            """
            A simple wrapper for your model

            Args:
                tokens: list of list of strings. The outer list represents the sentences, the inner one the tokens contained
                within it. Ex: [ ["This", "is", "the", "first", "homework"], ["Barack", "Obama", "was", "elected"] ]

            Returns:
                list of list of predictions associated to each token in the respective position.
                Ex: Ex: [ ["O", "O", "O", "O", "O"], ["PER", "PER", "O", "O"] ]

            """
            raise NotImplementedError
