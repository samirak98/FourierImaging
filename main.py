import torch
from utils import configuration as cf
from utils import datasets as data
from modules.perceptron import perceptron
import trainer

#%% Set up variable and data for an example
data_file = "../datasets/"
conf = cf.conf(data_file=data_file, download=True)

# get train, validation and test loader
train_loader, valid_loader, test_loader = data.load(conf)


#%% define the model and an instance of the best model class
sizes = [784, 200, 80, 10]
model = perceptron(sizes, conf.activation_function).to(conf.device)

#%% Initialize optimizer and lamda scheduler
opt = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
# initalize history
tracked = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
history = {key: [] for key in tracked}


def main():
    print("Train model: {}".format(conf.model))
    for i in range(conf.epochs):
        print(10*"<>")
        print(20*"|")
        print(10*"<>")
        print('Epoch', i)
        
        # train_step
        train_data = trainer.train_step(conf, model, opt, train_loader)
        
        # validation step
        val_data = trainer.validation_step(conf, model, valid_loader)
        
        # update history
        for key in tracked:
            if key in val_data:
                history[key].append(val_data[key])
            if key in train_data:
                history[key].append(train_data[key])
                
if __name__ == '__main__':
    main()