import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

#code modified from https://github.com/Cyanogenoid/pytorch-vqa/blob/master/model.py

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))

class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1, batch_first=True)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q):
        embedded = self.embedding(q)
        embedded = self.relu(self.drop(embedded))
        _, (_, c) = self.lstm(embedded)
        return c.squeeze(0)

class Net(nn.Module):
   def __init__(self, resnet_feature_size=1024, question_vocab_size=473, question_dim=300, lstm_feature_dim=1024, max_question_length=21, glimpses=2, attention_features=512, max_answer_length=35, answer_vocabulary_size=410):
       super(Net, self).__init__()
       self.text = TextProcessor(question_vocab_size, question_dim, lstm_feature_dim, 0.5)
       self.attention = Attention(resnet_feature_size, lstm_feature_dim, attention_features, glimpses, 0.5)
       self.max_answer_length = max_answer_length 
       self.answer_vocabulary_size = answer_vocabulary_size   
       self.answering = Classifier(glimpses * 1024 + lstm_feature_dim, 1024, max_answer_length*answer_vocabulary_size, 0.5)

  
   def forward(self, images, questions):
       questions = self.text(questions)
       images = images / (images.norm(p=2, dim=1, keepdim=True).expand_as(images) + 1e-8)
       a = self.attention(images, questions)
       images = apply_attention(images, a)
       combined = torch.cat([images, questions], dim=1)
       answer = self.answering(combined)
       answer = answer.view(-1, self.answer_vocabulary_size, self.max_answer_length) 
       return answer


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

#




def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled

