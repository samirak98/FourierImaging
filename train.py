import torch
import torch.nn.functional as F
#%% select loss function based on name
def select_loss(name):
    if name == 'L2':
        def loss_fct(x):
            return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)
    elif name == 'crossentropy':
        loss_fct = F.cross_entropy
    else:
        raise ValueError('Unknown loss function: ' + name)
        
    return loss_fct
#%% the trainer class
class trainer:
    def __init__(self, model, opt, train_loader, valid_loader, conf):
        self.model = model
        self.opt = opt
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss = select_loss(conf['loss'])
        self.device = conf['device']
        self.verbosity = conf['verbosity']
        

    def train_step(self):
        # train phase
        self.model.train()
        
        # initalize values for train accuracy and train loss
        train_acc = 0.0
        train_loss = 0.0
        tot_steps = 0
        
        # loop over all batches
        for batch_idx, (x, y) in enumerate(self.train_loader):
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
            tot_steps += y.shape[0]
        
        # print accuracy and loss
        if self.verbosity > 0: 
            print(50*"-")
            print('Train Accuracy:', train_acc/tot_steps)
            print('Train Loss:', train_loss)
        return {'train_loss':train_loss, 'train_acc':train_acc/tot_steps}



    def validation_step(self):
        val_acc = 0.0
        val_loss = 0.0
        tot_steps = 0
        
        # -------------------------------------------------------------------------
        # loop over all batches
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
            
        # print accuracy
        if self.verbosity > 0: 
            print(50*"-")
            print('Validation Accuracy:', val_acc/tot_steps)
        return {'val_loss':val_loss, 'val_acc':val_acc/tot_steps}

def test_step(conf, model, test_loader, attack = None, verbosity = 1):
    model.eval()
    
    test_acc = 0.0
    test_loss = 0.0
    tot_steps = 0
    
    if attack is None:
        attack = conf.attack
    # -------------------------------------------------------------------------
    # loop over all batches
    for batch_idx, (x, y) in enumerate(test_loader):
        # get batch data
        x, y = x.to(conf.device), y.to(conf.device)
        
         # evaluate model on batch
        logits = model(x)
        
        # Get classification loss
        loss = conf.loss(logits, y)
        
        test_acc += (logits.max(1)[1] == y).sum().item()
        test_loss += loss.item()
        tot_steps += y.shape[0]
        
    # print accuracy
    if verbosity > 0: 
        print(50*"-")
        print('Test Accuracy:', test_acc/tot_steps)
    return {'test_loss':test_loss, 'test_acc':test_acc/tot_steps}