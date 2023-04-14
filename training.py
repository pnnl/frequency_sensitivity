"""
Standard training algorithms.
"""
import torch
from torch import optim, nn
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os, re
from copy import deepcopy
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from constants import results_dir
from metrics import Acc
from typing import Callable
import pathlib as pa

def train(model: nn.Module, 
    train_dataloader, 
    eval_dataloader,
    optimizer:str = 'sgd', 
    decay:float=1e-4,
    momentum:float=0.9,
    lr:float=1e-3, 
    num_epochs: int = 1,
    patience:int=10,
    min_lr: float = 1e-6,
    disable_progress_bar:bool=False,
    save_weights:bool=False,
    save_interval:int = None, 
    loss_function:nn.Module=nn.CrossEntropyLoss(),
    acc_function: Callable = Acc(),
    device:torch.device or int=None,
    distributed:bool=False,
    rank:int=None,
    mlflow_track:bool=False,
    out_dir:str or pa.Path ='./output',
    **kwargs):
    """The classical training loop

    BREAKING CHANGE: no longer "distributes" model/dataloaders if distributed. Needs
    to be passed an already distributed model/dataloaders.

    :param model: The object that will be trained
    :type model: torch
    :param train_dataloader: The data that the model will use for training
    :type train_dataloader: torch.utils.data.Dataset
    :param eval_dataloader: The data that the model will use for validation
    :type eval_dataloader: torch.utils.data.Dataset
    :param optimizer: The optimizer function, defaults to SGD
    :type optimizer: str, optional
    :param decay: Amount of weight decay. Defaults to zero.
    :type decay: float, optional
    :param momentum: momentum for SGD
    :type momentum: float
    :param lr: The learning rate, defaults to 1e-3
    :type lr: float, optional
    :param num_epochs: The number of epochs to loop over, defaults to 10
    :type num_epochs: int, optional
    :param patience: how long to wait before dropping lr
    :type patience: int
    :param disable_progress_bar: Indicates whether or not the user wants to
        disable to progress bar shown when the model is running, defaults to
        False
    :type disable_progress_bar: bool, optional
    :param save_weights: Tells the model to save the weights or not, defaults
        to False
    :type save_weights: bool, optional
    :param save_interval: Indicates how frequently the model saves a weights
        file, defaults to None
    :type save_interval: int, optional
    :param loss_function: The loss function, defaults to None
    :type loss_function: function, optional
    :param device: gpu to use
    :type device: torch.device
    :param distributed: whether to use torch.distributed
    :type distributed: bool
    :param rank: if distributed, which gpu to target
    :type rank: int 
    :return: None
    :rtype: None
    """
    if mlflow_track:
        print('tracking w/ mlflow')
        import mlflow
    if distributed and device:
        raise RuntimeError(f'args distributed and device are mut. excl.')
    if (not distributed) and (device == None):
        device = torch.cuda.current_device()
    # Only print/save output in one of the distributed processes
    log_process = (not distributed) or (distributed and rank==0)
    if log_process:
        print(f'storing results in {out_dir}')
        if os.path.exists(out_dir) == False:
            pa.Path(out_dir).mkdir(parents=True)
        with open(os.path.join(out_dir, 'log.txt'), 'w') as f:
            # Low-tech way of printing a summary of training parameters 
            message = f'run of classical train with arguments:\n{locals()}\n'
            f.write(message)
            # message = f'train dataset summary: {train_dataloader.dataset.__str__()}\neval dataset summary: {eval_dataloader.dataset.__str__()}\n'
            # f.write(message)
            # message = f'model summary:\n{model.__str__()}\n'
            # f.write(message)
    # In the distributed case, set device = rank and ensure the model is on the
    # appropriate device 
    # actually, since we are assuming the model is already ddp-ed, don't push
    # the model?
    if distributed:
        device = rank
        # actually, since we are assuming the model is already ddp-ed, don't push
        # the model?
        # model.to(device)
    else:
        model.to(device)
    # In both optimizers below, we only apply weight decay to the weights :-)
    if optimizer=='sgd':
        optimizer = optim.SGD([
            {'params': [x[1] for x in model.named_parameters() if re.search('weight', x[0])!= None], 'weight_decay': decay},
            {'params': [x[1] for x in model.named_parameters() if re.search('weight', x[0]) == None], 'weight_decay': 0.0}
            ], lr=lr, momentum=momentum)
    elif optimizer == 'adam':
        optimizer = optim.Adam([
            {'params': [x[1] for x in model.named_parameters() if re.search('weight', x[0])!= None], 'weight_decay': decay},
            {'params': [x[1] for x in model.named_parameters() if re.search('weight', x[0]) == None], 'weight_decay': 0.0}
            ], lr=lr, betas=(0.9, 0.999))
    # drop lr by factor of 10 if val acc hasn't improved by 1% for last ``patience``
    # epochs. There could be a more optimal strategy, this was chosen arbitrarily
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, 
        patience=patience, threshold=0.01, threshold_mode='abs', min_lr=min_lr)
    # amp stuff
    scaler = GradScaler()
    # Helper values for keeping track of best weights 
    best_acc = 0.0
    best_wts = deepcopy(model.state_dict())
    # Keep track of learning curves 
    train_curve_loss = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    train_curve_acc = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    val_curve_acc = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    val_curve_loss = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    lr_curve = torch.zeros((num_epochs,), dtype=torch.float).to(device)
    # We need these to synchronize loss and accuracy between processes  
    if distributed:
        train_loss = torch.tensor(0.0).cuda(device)
        val_acc = torch.tensor(0.0).cuda(device)
        learn_rate = torch.tensor(0.0).cuda(device)    

    def save_stuff(out_dir, train_curve_acc, train_curve_loss, 
        val_curve_acc, val_curve_loss, 
        lr_curve, best_wts, mlflow_track:bool = False):
        # Make sure we only save from one distributed process (paranoia)
        if log_process:
            torch.save(train_curve_acc.cpu(), os.path.join(out_dir, 'train_curve_acc.pt'))
            torch.save(train_curve_loss.cpu(), os.path.join(out_dir, 'train_curve_loss.pt'))
            torch.save(val_curve_acc.cpu(), os.path.join(out_dir, 'val_curve_acc.pt'))
            torch.save(val_curve_loss.cpu(), os.path.join(out_dir, 'val_curve_loss.pt'))
            torch.save(lr_curve.cpu(), os.path.join(out_dir, 'lr_curve.pt'))
            torch.save(best_wts, os.path.join(out_dir, 'best_wts.pt'))
            final_wts = deepcopy(model.state_dict())
            torch.save(final_wts, os.path.join(out_dir, 'final_wts.pt'))
            msg = f'saved learning curves and weights to {out_dir}'
            print(msg)
            if mlflow_track:
                mlflow.log_artifact(os.path.join(out_dir, 'train_curve_acc.pt'))
                mlflow.log_artifact(os.path.join(out_dir, 'train_curve_loss.pt'))
                mlflow.log_artifact(os.path.join(out_dir, 'val_curve_acc.pt'))
                mlflow.log_artifact(os.path.join(out_dir, 'val_curve_loss.pt'))
                mlflow.log_artifact(os.path.join(out_dir, 'lr_curve.pt'))
                mlflow.log_artifact(os.path.join(out_dir, 'best_wts.pt'))
                # mlflow.log_artifact( os.path.join(out_dir, 'final_wts.pt'))

    # Outer epoch loop
    epoch_desc = f'Epoch'
    epoch_loop = range(num_epochs)
    # Only display progress from one distributed process 
    if (not disable_progress_bar) and log_process:
        epoch_loop = tqdm(epoch_loop, total=num_epochs, 
            desc=epoch_desc, disable=disable_progress_bar)
    for epoch_index in epoch_loop:
        spacer = ' ' * (5 - len(str(epoch_index)))
        train_desc = f'Training model - Iteration: ' \
                     f'{epoch_index}' + spacer
        eval_desc = f'Evaluating model - Iteration: ' \
                    f'{epoch_index}' + spacer
        # This is the start of the training loop
        train_gen = train_dataloader
        if (not disable_progress_bar) and log_process:
            train_gen = tqdm(
                    train_gen,
                    # total=len(train_dataloader),
                    desc=train_desc,
                    disable=disable_progress_bar
                )
        model.train()
        # In the distributed case, sync up samplers 
        if distributed and isinstance(train_dataloader, DataLoader):
            train_dataloader.sampler.set_epoch(epoch_index)
        for sample, labels in train_gen:
            optimizer.zero_grad()
            if type(sample) is dict:
                sample = {k: v.to(device) for k, v in sample.items()}
            else:
                sample = sample.to(device)
            labels = labels.to(device)
            with autocast():
                output = model(sample)
                # temporary fix to work with torch segmentation models
                if type(output) is not torch.Tensor:
                    output = output['out']
                loss = loss_function(output, labels)
                train_acc = acc_function(output, labels)
            # amp backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_curve_loss[epoch_index] += loss.item()
            train_curve_acc[epoch_index] += train_acc.item()
            if (not disable_progress_bar) and log_process:
                train_gen.set_description(f'{train_desc}, loss: {loss.item():.4g}, acc: {train_acc.item():.4g}')
        train_curve_loss[epoch_index] /= len(train_dataloader)
        train_curve_acc[epoch_index] /= len(train_dataloader)
        if (log_process and mlflow_track):
                mlflow.log_metric('train_acc', train_curve_acc[epoch_index], step=epoch_index)
                mlflow.log_metric('train_loss', train_curve_loss[epoch_index], step=epoch_index)
        # Perform the evaluation step of the training loop
        eval_gen = eval_dataloader
        if (not disable_progress_bar) and log_process:
            eval_gen = tqdm(
                    eval_gen,
                    # total=len(eval_dataloader),
                    desc=eval_desc,
                    disable=disable_progress_bar
                )
        model.eval()
        # In the distributed case, sync up samplers 
        if distributed and isinstance(train_dataloader, DataLoader):
            eval_dataloader.sampler.set_epoch(epoch_index)
        for sample, labels in eval_gen:
            with torch.no_grad():
                if type(sample) is dict:
                    sample = {k: v.to(device) for k, v in sample.items()}
                else:
                    sample = sample.to(device)
                labels = labels.to(device)
                with autocast():
                    output = model(sample)
                    if type(output) is not torch.Tensor:
                        output = output['out']                
                    loss = loss_function(output, labels)
                    eval_acc = acc_function(output, labels)
            val_curve_loss[epoch_index] += loss.item()
            val_curve_acc[epoch_index] += eval_acc
            if (not disable_progress_bar) and log_process:
                eval_gen.set_description(f'{eval_desc}, loss: {loss.item():.4g}, acc: {eval_acc.item():.4g}')
        val_curve_loss[epoch_index] /= len(eval_dataloader)
        val_curve_acc[epoch_index] /= len(eval_dataloader)

        lr_curve[epoch_index] = torch.tensor([p['lr'] for p in optimizer.param_groups]).mean()
        # Most complicated piece of distributed case: we need to synchronize
        # loss and accuracy across processes, to obtain the correct loss and
        # validation curves. 
        if distributed:
            train_loss = train_curve_loss[epoch_index]
            dist.all_reduce(train_loss.cuda(device), op = dist.ReduceOp.SUM)
            train_curve_loss[epoch_index] = train_loss/torch.cuda.device_count()
            val_acc = val_curve_acc[epoch_index]
            dist.all_reduce(val_acc.cuda(device), op=dist.ReduceOp.SUM)
            val_curve_acc[epoch_index] = val_acc/torch.cuda.device_count()
            lr_curve[epoch_index] = torch.tensor([p['lr'] for p in optimizer.param_groups]).mean()
            learn_rate = lr_curve[epoch_index]
            dist.all_reduce(learn_rate.cuda(device), op=dist.ReduceOp.SUM)
            lr_curve[epoch_index] = learn_rate/torch.cuda.device_count()
        # Check if current validation accuracy is the best so far (if
        # distributed, only need to do this in the logging process)
        if log_process:
            if val_curve_acc[epoch_index] > best_acc:
                # If so, update best accuracy and best weights 
                best_acc = val_curve_acc[epoch_index]
                best_wts = deepcopy(model.state_dict())
                better_val_loss = True
                if mlflow_track:
                    mlflow.log_metric('best_val_acc', best_acc)
                    mlflow.log_metric('best_epoch', epoch_index)
            else:
                better_val_loss = False
        # Test reduce on plateau criterion, step the lr scheduler. I *think*
        # this should be done in all processes (based on examples at
        # https://github.com/pytorch/examples/blob/main/imagenet/main.py, but there
        # is literally no doc on the topic)
        scheduler.step(val_curve_acc[epoch_index])
        # Save weights at specified interval and if we just hit a new best
        if (save_weights and log_process):
            if (epoch_index % save_interval == 0) or better_val_loss:
                save_stuff(out_dir, train_curve_acc, train_curve_loss, val_curve_acc, val_curve_loss, lr_curve, best_wts, mlflow_track)
        # print info
        if log_process:
            msg =  f'{epoch_desc}: {epoch_index}, train loss: {train_curve_loss[epoch_index]:.4g}, ' \
                + f'val acc: {val_curve_acc[epoch_index]:.4g}, better? {better_val_loss:.4g}, lr: {lr_curve[epoch_index]:.4g}'
            if not disable_progress_bar:
                epoch_loop.set_description(msg)
            else:
                print(msg)
            if mlflow_track:
                mlflow.log_metric('val_acc', val_curve_acc[epoch_index], step=epoch_index)
                mlflow.log_metric('val_loss', val_curve_loss[epoch_index], step=epoch_index)
                mlflow.log_metric('learning_rate_epoch', lr, step=epoch_index)
        # for some reason this isn't triggering...
        if torch.allclose(lr_curve[epoch_index].to(device), torch.tensor([l for l in scheduler.min_lrs]).mean().to(device)):
            save_stuff(out_dir, train_curve_acc, train_curve_loss, val_curve_acc, val_curve_loss, lr_curve, best_wts, mlflow_track)
            if log_process:
                err_msg = f'hit minimum lr of {torch.tensor([l for l in scheduler.min_lrs]).mean():.4g}'
                err_msg = err_msg + f'best val acc: {val_curve_acc.max().item():.4g}'
                print(err_msg)
            break  

    # ensure we save after loop ends ...
    if (save_weights and log_process):
        save_stuff(out_dir, train_curve_acc, train_curve_loss, val_curve_acc, val_curve_loss, lr_curve, best_wts, mlflow_track)
    # Load up the best weights, in case this is being used interactively 
    # only makes sense if not distributed
    if not distributed:
        model.load_state_dict(best_wts)
    return None