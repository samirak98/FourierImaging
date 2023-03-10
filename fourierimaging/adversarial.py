#copied from CLIP (https://github.com/TimRoith/CLIP/blob/main/adversarial_attacks.py)

import torch

class attack:
    pass

# No attack     
class no_attack(attack):
    #
    def __call__(self, model, x, y):
        return x

# fgsm attack
class fgsm(attack):
    def __init__(self, loss, epsilon=0.3, x_min=0.0, x_max=1.0):
        super(fgsm, self).__init__()
        self.epsilon = epsilon
        self.loss = loss
        self.x_min = x_min
        self.x_max = x_max
        
    def __call__(self, model, x, y):
        #get delta
        delta = get_delta(x, self.epsilon, x_min=self.x_min, x_max=self.x_max)
        delta.requires_grad = True
        # get loss
        pred = model(x + delta)
        loss = self.loss(pred, y)
        loss.backward()
        # get example
        grad = delta.grad.detach()
        delta.data = delta + self.epsilon * torch.sign(grad)
        return torch.clamp(x + delta.detach(), min=self.x_min, max=self.x_max)
    
def clamp(x, x_min, x_max):
    return torch.max(torch.min(x, x_max), x_min)
                    
def get_delta(x, eps=1.0, uniform=False, x_min=0.0, x_max=1.0):
    delta = torch.zeros_like(x)
    if uniform:
        delta.uniform_(-eps, eps)
    return clamp(delta, x_min - x, x_max - x)