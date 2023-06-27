from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
from torch.utils.data.datapipes import iter
import math
import torch.nn as nn 
from functools import partial
from ray.air import session
import numpy as np
import os
import matplotlib.pyplot as plt 

from os.path import isfile, join

 

path = './data'
def clean_and_split_data(data_set):
    data_set = data_set.split(' ')
    data_set = list(filter(lambda x: x != '', data_set))
    return data_set

def encode_data(data, char2index):
  data = [char2index[c] for c in data]
  return data 

def decode_data(data, index2char):
  data = [index2char[c] for c in data]
  return data 


def load_data(path):

  with open(join(path,'ptb.train.txt')) as f:
      train_data = f.read()
      
  with open(join(path,'ptb.valid.txt')) as f:
      val_data = f.read()
      
  with open(join(path,'ptb.test.txt')) as f:
      test_data = f.read()
       

  ## clean and tokenize 
  train_data =clean_and_split_data(train_data)
  val_data =clean_and_split_data(val_data)  
  test_data =clean_and_split_data(test_data)

  ## vocab
  words = sorted(set(train_data+val_data+test_data))

  ##encoder
  char2index = {c: i for i, c in enumerate(words)}
  ##decoder 
  index2char = {i: c for i, c in enumerate(words)}



  ## encode 
  train_data = encode_data(train_data,char2index)
  val_data = encode_data(val_data,char2index)
  test_data = encode_data(test_data,char2index)
  
  return train_data, val_data, test_data, index2char,words
def make_batchs(data,batch_size):
  try :
    data = iter.Batcher(data,batch_size = batch_size, drop_last=True)
  except:
    print('needs to get an itrable Wraped data - make seq first')

  return data 

def make_seq(data,seq_length):
  label = iter.IterableWrapper(data[1:])
  data = iter.IterableWrapper(data)
   
  label = iter.Batcher(label,batch_size = seq_length, drop_last=True)
  data = iter.Batcher(data,batch_size = seq_length, drop_last=True)

  return data,label
  
class wordModel(nn.Module):
    
    def __init__(self,words,device, num_hidden=200, num_layers=2,drop_prob=0.5,with_drop = False):
        
        
        # SET UP ATTRIBUTES
        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.words = words
        self.with_drop = with_drop
        
      
      
        self.lstm = nn.LSTM(num_hidden, num_hidden, num_layers, dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc_linear = nn.Linear(num_hidden, len(self.words))
      
        self.embeding = nn.Embedding(len(words),self.num_hidden)

        self.device = device  


    def forward(self, x, hidden):
                  
        x = self.embeding(x)
       
        lstm_output, hidden = self.lstm(x, hidden)
        
        if self.with_drop:
          drop_output = self.dropout(lstm_output)
          
          drop_output = drop_output.contiguous().view(-1, self.num_hidden)
          final_out = self.fc_linear(drop_output)

        else:
          out = lstm_output.contiguous().view(-1, self.num_hidden)
          final_out = self.fc_linear(out)
        
        
        return final_out, hidden
    
    
    def hidden_state(self, batch_size):

        if self.device == 'cuda':
            
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda(),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda())
        else:
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden))
        
    
        return hidden
def plot_graph(num_epochs, training_perplexity, validation_perplexity, dropout=False):
    x = [el+1 for el in range(num_epochs)]
    y_1 = training_perplexity
    y_2 = validation_perplexity
    title = None
    if dropout:
        title = 'Dropout'
    else:
        title = 'No Dropout'
    plt.plot(x, y_1, color='blue')
    plt.plot(x, y_2, color='orange')
    plt.xlabel('Epoch number')
    plt.ylabel('Perplexity')
    plt.legend(['training perplexity', 'validation perplexity'])
    plt.title(title)
    plt.show()
    print(f'best training perplexity was {min(training_perplexity)}')
    print(f'best testing perplexity was {min(validation_perplexity)}')


def calculate_perplexity(losses):
    total_loss = sum(losses)
    average_loss = total_loss / len(losses)
    perplexity = math.exp(average_loss)
    return perplexity

def train_eval_lstm(config,checkpoint_dir=None):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  path = r'/home/robotics20/Documents/rotem/learn/deep/data/data'

  ## read file and split 
  train_data, val_data, test_data, index2char,words = load_data(path) 
  ## make seq and batches 
  train_data,train_label = make_seq(train_data,config["seq_length"])

  train_data = make_batchs(train_data,config["batch_size"])
  train_label = make_batchs(train_label,config["batch_size"])


  val_data,val_label = make_seq(val_data,config["seq_length"])

  val_data = make_batchs(val_data,config["batch_size"])
  val_label = make_batchs(val_label,config["batch_size"])


  ### to tensors 
  train_data = torch.tensor(np.array(train_data),dtype=torch.int)
  train_label = torch.tensor(np.array(train_label),dtype=torch.int)


  val_data = torch.tensor(np.array(val_data),dtype=torch.int)
  val_label = torch.tensor(np.array(val_label),dtype=torch.int)

  model = wordModel(
    words = words,
    device = device,
    num_hidden=config["num_hidden"],
    num_layers=config["num_hidden"],
    drop_prob=0.5,
    with_drop=False,
  )

  model = model.to(device)


  optimizer = torch.optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"],weight_decay=config["weight_decay"])
  criterion = nn.CrossEntropyLoss()
  schedualer = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=config["gamma"])

  train_losses = []
  val_losses = []
  train_preplexity = []
  val_preplexity = []
  print("starts training ")

  for i in range(int(config["epoch"])):
    model.train()
    hidden = model.hidden_state(batch_size=config["batch_size"])
    print(i)
    
    for j, featurs in enumerate(train_data):
      labels = train_label[j]
      featurs = featurs.to(device)
      labels = labels.to(device)
      
      hidden = tuple([state.data for state in hidden])    
      lstm_output, hidden = model.forward(featurs,hidden)
      
      loss = criterion(lstm_output,labels.view(config["batch_size"]*config["seq_length"]).long())
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)

      optimizer.step()
      optimizer.zero_grad()
      
    schedualer.step()
    ### validation 
    model.eval()
    val_hidden = model.hidden_state(config["batch_size"])
    for k, val_featurs in enumerate(val_data):
      val_labels = val_label[k]
      val_featurs = val_featurs.to(device)
      val_labels = val_labels.to(device)

      val_hidden = tuple([state.data for state in val_hidden])   
      lstm_output, val_hidden = model.forward(val_featurs,val_hidden)
      val_loss = criterion(lstm_output,val_labels.view(config["batch_size"]*config["seq_length"]).long())

    model.train()


    train_losses.append(loss.item())
    #preplexcity
    train_preplexity.append(calculate_perplexity(train_losses)) 
    val_losses.append(val_loss.item())
    #preplexcity
    val_preplexity.append(calculate_perplexity(val_losses))
    print(f"Epoch: {i} Step: {j} trainloss: {loss.item()} Val Loss: {val_loss.item()}")
   
    with tune.checkpoint_dir(i) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((model.state_dict(), optimizer.state_dict()), path)

    tune.report(loss=val_loss.item())
  return train_preplexity, val_preplexity

max_num_epochs = 20
num_samples = 5
config = {
    "gamma": tune.loguniform(0.9, 0.99),
    "weight_decay": tune.loguniform(0.0001,0.001 ),
    "lr": tune.loguniform(1, 6),
    "momentum": tune.loguniform(0.5, 0.9),
    "num_hidden": tune.choice([200,]),
    "num_layers": tune.choice([2]),
    "seq_length": tune.choice([20]),
    "batch_size": tune.choice([10, 20, 30, 40,50]),
    "epoch": tune.uniform(5,5)
}
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2)
reporter = CLIReporter(
    # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
    metric_columns=["loss","val_preplexity","epoch"])
result = tune.run(
    partial(train_eval_lstm ),
    resources_per_trial={"cpu": 20, "gpu": 1},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter)
best_trial = result.get_best_trial("val_loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["val_loss"]))
print("Best trial final val_preplexity: {}".format(
    best_trial.last_result["val_preplexity"]))