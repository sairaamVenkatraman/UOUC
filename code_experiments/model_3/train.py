import numpy as np
import json
import os

import types
import torch
import numpy as np
import collections

from random import shuffle
import random



import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

from resnet import *
from data_loader import *
from model import *

from torchtext.data.metrics import bleu_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load train dictionary
with open('train_question_dictionary.json', 'r') as fp:
     train_question_dictionary = json.load(fp)
fp.close()
#load test 1 dictionary
with open('test_1_question_dictionary.json', 'r') as fp:
     test_1_question_dictionary = json.load(fp)
fp.close()
#load test 2 dictionary
with open('test_2_question_dictionary.json', 'r') as fp:
     test_2_question_dictionary = json.load(fp)
fp.close()
#load test 3 dictionary
with open('test_3_question_dictionary.json', 'r') as fp:
     test_3_question_dictionary = json.load(fp)
fp.close()


#given a index return a torch tensor of questions and answers
def question_answer_load(split, index):
    global train_question_dictionary
    global test_1_question_dictionary
    global test_2_question_dictionary
    global test_3_question_dictionary
    #get train
    if split == 'train':
       questions = train_question_dictionary[index]
              
    #get test 1
    if split == 'test_1':
       questions = test_1_question_dictionary[index]
           
    #get test 2
    if split == 'test_2':
       questions = test_2_question_dictionary[index]
       
    #get test 3
    if split == 'test_3':
       questions = test_3_question_dictionary[index]
       
    
    number_of_question = len(questions.keys())
    question_index = random.randint(1, number_of_question)
    questions = questions[str(question_index)]
    question = questions[0]
    answer = questions[1]
    question = torch.from_numpy(np.array(question))
    answer = torch.from_numpy(np.array(answer)) 
    return question, answer


#create a padded tensor when given a list of indices
def get_question_answer(split, indices):
    QUESTION_MAXIMUM_LENGTH = 21
    ANSWER_MAXIMUM_LENGTH = 35
    questions = []
    answers = []
    for index in indices:
        question, answer = question_answer_load(split, str(index.item()))
        questions.append(question)
        answers.append(answer)
    #pad with end of token 1
    question = torch.ones(indices.shape[0], QUESTION_MAXIMUM_LENGTH)
    questions = torch.nn.utils.rnn.pad_sequence(questions, True, 1)
    question[:, :questions.size(1)] = questions
    #pad with end of token 1
    answer = torch.ones(indices.shape[0], ANSWER_MAXIMUM_LENGTH)
    answers = torch.nn.utils.rnn.pad_sequence(answers, True, 1)
    answer[:, :answers.size(1)] = answers   
    return question.long(), answer.long()

#create training loader
train_set = TrainPipe(256, 4, 0)
train_set.build()
train_loader = DALIGenericIterator(train_set, ['data', 'label'], reader_name='Reader')

#create test 1 loader
test_1_set = TestPipeline(256, 4, 0)
test_1_set.build()
test_1_loader = DALIGenericIterator(test_1_set, ['data', 'label'], reader_name='Reader_1')

#load resnet model
resnet_50 = nn.DataParallel(resnet50().to(device))
resnet_50.load_state_dict(torch.load('checkpoint/ckpt.pth')['model'])
for parameter in resnet_50.parameters():
    parameter.requires_grad = False

#define model
INPUT_DIM = 473
OUTPUT_DIM = 410
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)
              
net = nn.DataParallel(Net(enc, dec, 1, 1, device).to(device))
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
net.apply(initialize_weights);
num_epochs = 50
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(net.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = 1)
#optimizer
"""optimizer =  torch.optim.AdamW(net.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, epochs=50, steps_per_epoch=200000//256)
criterion = nn.CrossEntropyLoss()"""


#define train function
def train(epoch):
    print('Epoch ', epoch)
    net.train()
    total = 0.0
    train_loss = 0.0
    i = 0
    for batch_idx, data in enumerate(train_loader):
        images = data[0]['data'].to(device)
        labels = data[0]['label']
        questions, answers = get_question_answer('train', labels.long())
        #print(questions.size())
        #print(answers.size())
        i = batch_idx
        optimizer.zero_grad()
        images = resnet_50(images).mean(1).view(-1, 1, 14*14)
        questions = questions.to(device)
        answers = answers.to(device)
        predicted_answers, _ = net(questions, answers, images)
        predicted_answers = predicted_answers.permute(0,2,1).contiguous()
        #print(predicted_answers.size())
        #print(answers.size())
        #assert False
        #output_dim = predicted_answers.shape[-1]            
        #predicted_answers = predicted_answers.contiguous().view(-1, output_dim)
        #answers = answers[:,1:].contiguous().view(-1)
        
        
        loss = criterion(predicted_answers, answers.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        #scheduler.step()
        #print(loss.item())
        total += questions.size(0)
        train_loss += loss.item()
        if batch_idx%100 == 0:
           print(train_loss/(batch_idx+1))
    train_loader.reset()
    print('Saving..')
    state = {
             'model': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'train_loss': train_loss/(i+1),
            }
    torch.save(state, 'checkpoint_1/checkpoint'+' '+str(epoch)+'.pth')

#define test function
def test(epoch):
    print('Epoch ', epoch)
    net.eval()
    total = 0.0
    test_1_loss = 0.0
    i = 0
    with torch.no_grad():
         for batch_idx, data in enumerate(test_1_loader):
             images = data[0]['data'].to(device)
             labels = data[0]['label']
             questions, answers = get_question_answer('test_1', labels.long())
             #print(questions.size())
             #print(answers.size())
             i = batch_idx
             #optimizer.zero_grad()
             images = resnet_50(images).mean(1).view(-1, 1, 14*14)
             questions = questions.to(device)
             answers = answers.to(device)
             predicted_answers, _ = net(questions, answers, images)
             predicted_answers = predicted_answers.permute(0,2,1).contiguous()
             #print(predicted_answers.size())
             #print(answers)         
            
             #output_dim = predicted_answers.shape[-1]            
             #predicted_answers = predicted_answers.contiguous().view(-1, output_dim)
             #answers = answers[:,1:].contiguous().view(-1)

             loss = criterion(predicted_answers, answers.long())
             total += questions.size(0)
             test_1_loss += loss.item()
             #print(loss.item())
             if batch_idx%10 == 0:
                print(test_1_loss/(batch_idx+1))
         test_1_loader.reset()
         state = torch.load('checkpoint_1/checkpoint'+' '+str(epoch)+'.pth')
         state['test_1_loss'] = test_1_loss/(i+1)
         torch.save(state, 'checkpoint_1/checkpoint'+' '+str(epoch)+'.pth')
         


for epoch in range(num_epochs):
    train(epoch)
    test(epoch)
