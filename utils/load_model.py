import torch
from modules.perceptron import perceptron

def load_model(conf):
    if conf['type'] == 'perceptron':
        model = perceptron(conf['sizes'], conf['activation_function'])
    else:
        raise ValueError('Unknown model type: ' + conf['type'])
    
    return model