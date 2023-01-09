def train_step(conf, model, opt, train_loader, verbosity = 1):
    # train phase
    model.train()
    
    # initalize values for train accuracy and train loss
    train_acc = 0.0
    train_loss = 0.0
    tot_steps = 0
    
    # -------------------------------------------------------------------------
    # loop over all batches
    for batch_idx, (x, y) in enumerate(train_loader):
        # get batch data
        x, y = x.to(conf.device), y.to(conf.device)
        
        # reset gradients
        opt.zero_grad()
        
        # evaluate model on batch
        logits = model(x)
        
        # Get classification loss
        loss = conf.loss(logits, y)
        
        # Update model parameters
        loss.backward()
        opt.step()
        
        # update accurcy and loss
        train_acc += (logits.max(1)[1] == y).sum().item()
        train_loss += loss.item()
        tot_steps += y.shape[0]
    
    # print accuracy and loss
    if verbosity > 0: 
        print(50*"-")
        print('Train Accuracy:', train_acc/tot_steps)
        print('Train Loss:', train_loss)
    return {'train_loss':train_loss, 'train_acc':train_acc/tot_steps}



def validation_step(conf, model, validation_loader, verbosity = 1):
    val_acc = 0.0
    val_loss = 0.0
    tot_steps = 0
    
    # -------------------------------------------------------------------------
    # loop over all batches
    for batch_idx, (x, y) in enumerate(validation_loader):
        # get batch data
        x, y = x.to(conf.device), y.to(conf.device)
        
         # evaluate model on batch
        logits = model(x)
        
        # Get classification loss
        loss = conf.loss(logits, y)
        
        val_acc += (logits.max(1)[1] == y).sum().item()
        val_loss += loss.item()
        tot_steps += y.shape[0]
        
    # print accuracy
    if verbosity > 0: 
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