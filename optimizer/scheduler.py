import torch
import numpy as np
import matplotlib.pyplot as plt

def lr_policy(step, epoch, initial_lr, optimizer, steps_per_epoch, warmup_epochs,
              hold_epochs, num_epochs=None, policy='linear', min_lr=1e-5,
              exp_gamma=None):
    """
    learning rate decay
    Args:
        initial_lr: base learning rate
        step: current iteration number
        N: total number of iterations over which learning rate is decayed
        lr_steps: list of steps to apply exp_gamma
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    hold_steps = hold_epochs * steps_per_epoch

    if policy == 'legacy':
        assert num_epochs is not None
        tot_steps = num_epochs * steps_per_epoch

        if step < warmup_steps:
            a = (step + 1) / (warmup_steps + 1)
        elif step < warmup_steps + hold_steps:
            a = 1.0
        else:
            a = (((tot_steps - step)
                 / (tot_steps - warmup_steps - hold_steps)) ** 2)

    elif policy == 'exponential':
        assert exp_gamma is not None

        if step < warmup_steps:
            a = (step + 1) / (warmup_steps + 1)
        elif step < warmup_steps + hold_steps:
            a = 1.0
        else:
            a = exp_gamma ** (epoch - warmup_epochs - hold_epochs)

    else:
        raise ValueError

    new_lr = max(a * initial_lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def decay_learning_rate(init_lr, i, iter_per_epoch, start_epoch, warm_up):
    warmup_threshold = warm_up
    step = start_epoch * iter_per_epoch + i + 1
    decayed_lr = init_lr * warmup_threshold ** 0.5 * min(step * warmup_threshold**-1.5, step**-0.5)
    return decayed_lr
    
class ScheduledOptim():
    """
    ref: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    """
    def __init__(self, d_model, n_warmup_steps):
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        lr = self._update_learning_rate()
        return lr        

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        
        return lr


dataset_len = 1000000
batch_size = 64

init_lr = 1e-3
iter_per_epoch = dataset_len // batch_size
num_epochs = 100
warm_up = 4000

mode = 'schedule'

if mode == 'schedule':
    my_optim = ScheduledOptim(d_model=512, n_warmup_steps = warm_up)

#elif mode == 'decay':

lr_list = list()
for epoch in range(num_epochs):    
    
    for i in range(iter_per_epoch):
        if mode == 'schedule':
            lr = my_optim.step_and_update_lr()
        elif mode == 'decay':
            lr = decay_learning_rate(init_lr, i, iter_per_epoch, epoch, warm_up)
        lr_list.append(lr)

plt.figure()
plt.plot(np.arange(len(lr_list)), lr_list)
plt.show() 