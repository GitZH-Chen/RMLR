import datetime
import geoopt
import time
import torch as th

import random
import numpy as np
from omegaconf import DictConfig, OmegaConf

from datasets.spdnet.HDM05_Loader import DataLoaderHDM05 as DataLoaderHDM05_spd

from Geometry.spd.spd_matrices import tril_param_metric,bi_param_metric,single_param_metric

def set_seed_only(seed):
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def set_seed_thread(seed,threadnum):
    th.set_num_threads(threadnum)
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def set_up(args):
    set_seed_thread(args.seed, args.threadnum)
    print('begin model {}'.format(args.modelname))
    print('writer path {}'.format(args.writer_path))

def get_dataset_settings(args):
    if args.dataset=='HDM05_SPD':
        pval = 0.5
        DataLoader = DataLoaderHDM05_spd(args.path, pval, args.batch_size)
    else:
        raise Exception('unknown dataset {}'.format(args.dataset))
    return DataLoader

def get_model_name(args):

    if args.classifier == 'SPDMLR':
        if args.metric in tril_param_metric:
            description = f'{args.metric}-[{args.power},{args.alpha},{args.beta:.4f}]'
        elif args.metric in bi_param_metric:
            description = f'{args.metric}-[{args.alpha},{args.beta:.4f}]'
        elif args.metric in single_param_metric:
            description = f'{args.metric}-[{args.power}]'
        description = '-' + description
    elif args.classifier == 'LogEigMLR':
        description=''

    description=f'{description}-{args.architecture}'

    optim = f'{args.lr}-{args.optimizer_mode}-{args.weight_decay}'
    name = f'{args.seed}-{optim}-{args.model_type}-{args.classifier}{description}-{datetime.datetime.now().strftime("%H_%M")}'

    return name

def optimzer(parameters,lr,mode='AMSGRAD',weight_decay=0.):
    if mode=='ADAM':
        optim = geoopt.optim.RiemannianAdam(parameters, lr=lr,weight_decay=weight_decay)
    elif mode=='SGD':
        optim = geoopt.optim.RiemannianSGD(parameters, lr=lr,weight_decay=weight_decay)
    elif mode=='AMSGRAD':
        optim = geoopt.optim.RiemannianAdam(parameters, lr=lr,amsgrad=True,weight_decay=weight_decay)
    else:
        raise Exception('unknown optimizer {}'.format(mode))
    return optim

def parse_cfg(args, cfg: DictConfig):
    # Function to recursively set attributes, keeping only the final key name
    def set_attributes_from_dict(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                # If the value is a dict, continue to extract its values
                set_attributes_from_dict(target, value)
            else:
                # Directly set the attribute on the target
                setattr(target, key, value)

    # Convert Hydra config to a nested dictionary and then flatten it
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    set_attributes_from_dict(args, cfg_dict)

    if cfg.fit.device != 'cpu' and th.cuda.is_available():
        args.device = th.device(f"cuda:{cfg.fit.device}")
    else:
        args.device = "cpu"


    # get model name
    args.modelname = get_model_name(args)
    return args

def train_per_epoch(model,args):
    start = time.time()
    epoch_loss, epoch_acc = [], []
    model.train()
    for local_batch, local_labels in args.DataLoader._train_generator:
        i = i + 1
        local_batch = local_batch.to(th.double).to(args.device)
        local_labels = local_labels.to(args.device)
        args.opti.zero_grad()
        out = model(local_batch)
        l = args.loss_fn(out, local_labels)
        acc, loss = (out.argmax(1) == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
        epoch_loss.append(loss)
        epoch_acc.append(acc)
        l.backward()

        args.opti.step()

    end = time.time()
    elapse = end - start
    return elapse,epoch_loss,epoch_acc

def val_per_epoch(model,args):
    epoch_loss, epoch_acc = [], []
    y_true, y_pred = [], []
    model.eval()
    with th.no_grad():
        for local_batch, local_labels in args.DataLoader._test_generator:
            local_batch = local_batch.to(th.double).to(args.device)
            local_labels = local_labels.to(args.device)
            out = model(local_batch)
            l = args.loss_fn(out, local_labels)
            predicted_labels = out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy()))
            y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            epoch_acc.append(acc)
            epoch_loss.append(loss)
    return epoch_loss,epoch_acc

def print_results(logger,training_time,acc_val,loss_val,epoch,args):
    if epoch % args.cycle == 0:
        logger.info(f'Time: {training_time[epoch]:.2f}, Val acc: {acc_val[epoch]:.2f}, loss: {loss_val[epoch]:.2f} at epoch {epoch + 1:d}/{args.epochs:d}')