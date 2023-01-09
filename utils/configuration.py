import torch
import torch.nn.functional as F
#import adversarial_attacks as at
#import models

#%% class for experiment configuration
class conf:
    def __init__(self, **kwargs):
        # model
        self.model = kwargs.get('model', "fc")
        self.activation_function = kwargs.get('activation_function', "ReLU")
        
        # dataset
        self.data_set = kwargs.get('data_set', "MNIST")
        self.data_set_mean = 0.0
        self.data_set_std = 1.0
        self.data_file = kwargs.get('data_file', "data")
        self.train_split = kwargs.get('train_split', 0.9)
        self.download = kwargs.get('download', False)
        self.im_shape = None
        self.x_min = 0.0
        self.x_max = 1.0
        
        # CUDA settings
        self.use_cuda = kwargs.get('use_cuda', False)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_workers = kwargs.get('num_workers', 1)
        
        # Loss function and norm
        def l2_norm(x):
            return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)
        self.loss = kwargs.get('loss', F.cross_entropy)
        self.in_norm = kwargs.get('in_norm', l2_norm)
        self.out_norm = kwargs.get('out_norm', l2_norm)
 
        # specification for Training
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 128)
        self.lr = kwargs.get('lr', 0.1)
        