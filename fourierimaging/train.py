import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

#%% select loss function based on name
def select_loss(name):
    if name == 'L2':
        def loss_fct(x):
            return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)
    elif name == 'crossentropy':
        #loss_fct = F.cross_entropy#
        loss_fct = nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function: ' + name)
        
    return loss_fct
    
#%% the trainer class
class trainer:
    def __init__(self, model, opt, lr_scheduler, train_loader, valid_loader, conf):
        self.model = model
        self.opt = opt
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss = select_loss(conf['loss'])
        self.device = conf['device']
        self.verbosity = conf['verbosity']
        self.lr_scheduler = lr_scheduler

    def train_step(self):
        # train phase
        self.model.train()
        
        # initalize values for train accuracy and train loss
        train_acc = 0.0
        train_loss = 0.0
        sample_ctr = 0
        epoch_ctr = 0
        
        # loop over all batches
        for batch_idx, (x, y) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            # get batch data
            x, y = x.to(self.device), y.to(self.device)
            
            self.opt.zero_grad() # reset gradients
            
            logits = self.model(x) # evaluate model on batch
            
            
            loss = self.loss(logits, y) # Get classification loss
            
            # Update model parameters
            loss.backward()
            self.opt.step()
            
            # update accurcy and loss
            train_acc += (logits.max(1)[1] == y).sum().item()
            train_loss += loss.item()
            sample_ctr += y.shape[0]
            epoch_ctr+=1
        if not (self.lr_scheduler is None):
            self.lr_scheduler.step(train_loss)
        # print accuracy and loss
        if self.verbosity > 0: 
            print(50*"-")
            print('Train accuracy: ' + str(100*train_acc/sample_ctr) +'[%]')
            print('Train loss:', train_loss/epoch_ctr)
        return {'train_loss':train_loss, 'train_acc':train_acc/sample_ctr}



    def validation_step(self):
        val_acc = 0.0
        val_loss = 0.0
        tot_steps = 0
        epoch_ctr = 0
        
        # loop over all batches
        if self.valid_loader is None:
            return {}
        else:  
            for batch_idx, (x, y) in enumerate(self.valid_loader):
                # get batch data
                x, y = x.to(self.device), y.to(self.device)
                
                # evaluate model on batch
                logits = self.model(x)
                
                # Get classification loss
                loss = self.loss(logits, y)
                
                val_acc += (logits.max(1)[1] == y).sum().item()
                val_loss += loss.item()
                tot_steps += y.shape[0]
                epoch_ctr += 1
                
            # print accuracy
            if self.verbosity > 0: 
                print(50*"-")
                print('Validation Accuracy: ' + str(100 * val_acc/tot_steps)+'[%]')
                print('Validation loss:', val_loss/epoch_ctr)
            return {'val_loss':val_loss, 'val_acc':val_acc/tot_steps}

class Tester:
    def __init__(self, test_loader, conf, attack = None):
        self.test_loader = test_loader
        self.loss = select_loss(conf['loss'])
        self.device = conf['device']
        self.verbosity = conf['verbosity']
        self.attack = attack

    def __call__(self, model):
        model.eval()
        test_acc = 0.0
        test_loss = 0.0
        tot_steps = 0
        epoch_ctr = 0
        # loop over all batches
        for batch_idx, (x, y) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            # get batch data
            x, y = x.to(self.device), y.to(self.device)

            # update x to an adversarial example (optional)
            if self.attack is not None:
                x = self.attack(model, x, y)
            
            # evaluate model on batch
            logits = model(x)
            
            # Get classification loss
            loss = self.loss(logits, y)
            
            test_acc += (logits.max(1)[1] == y).sum().item()
            test_loss += loss.item()
            tot_steps += y.shape[0]
            epoch_ctr += 1
            
        # print accuracy
        if self.verbosity > 0: 
            print(50*"-")
            print('Test Accuracy: ' + str(100*test_acc/tot_steps) + '[%]')
        return {'test_loss':test_loss/epoch_ctr, 'test_acc':test_acc/tot_steps}