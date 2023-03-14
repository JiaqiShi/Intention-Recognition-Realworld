from utils.json_config import JsonConfig
import torch
import os
from functools import reduce
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

import pickle

from utils.dataloader import *

from utils.train_eval_model import EarlyStopping, Train_Eval_Model, get_param_num
from utils.loss_function import Multi_weighted_crossentropyloss

from model.intentgcn import IntentGCN
from model.intention_models import GCN_and_MLP, GCN_MLP_MLP

DATALOADER = {'all-split-day':get_intention_dataloader_day}

MODEL = {'gcn':IntentGCN ,'gcn+mlp':GCN_and_MLP, 'gcn+mlp+mlp':GCN_MLP_MLP}

def main(args):

    # device
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:"+str(args.Data.device_index))
        else:
            device = torch.device("cuda:0") 
        print('[Info] Device: {}'.format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print('[Info] Device: {}'.format(device))

    # save path
    model_path = os.path.join('results_all', reduce(lambda x, y: x+'+'+y, args.Data.features))

    his_path = os.path.join(model_path, args.Model.model + '0')
    i = 0
    while True:
        if os.path.exists(his_path):
            i = i + 1
            his_path = os.path.join(model_path, args.Model.model + str(i))
        else:
            os.makedirs(his_path)
            print('[Info] History path is {}'.format(his_path))
            break

    # load data
    Dataloader_func = DATALOADER[args.Data.dataset]
    train_set, val_set, test_set, train_loader, val_loader, test_loader = Dataloader_func(args.Data.dir, args.Data.features, \
                                    args.Data.labels, batch_size=args.Optim.batch_size, cut_frame=args.Data.cut_frame)

    # model init
    Model = MODEL[args.Model.model]

    input_shape = train_set.get_shape()

    if args.is_attr_available('Graph_paras'):
        graph_args = {k:v for k,v in args.Graph_paras.items() if k != '__name'}
    else:
        graph_args = None

    if args.is_attr_available('GCN_paras'):
        GCN_paras = {k:v for k,v in args.GCN_paras.items() if k != '__name'}
    else:
        GCN_paras = {}

    if args.Model.model in ['gcn']:
        model = Model(input_shape, args.Model.out_channels, graph_args, **GCN_paras)
    elif args.Model.model in ['gcn+mlp', 'gcn+mlp+mlp']:
        model = Model(input_shape, args.Model.out_channels, graph_args, load_dict_paths=args.Model.sub_model_paths, is_detach=args.Model.sub_model_detach, device=device, **GCN_paras)
    else:
        model = Model(input_shape, args.Model.out_channels)

    best_model_path = os.path.join(his_path, 'model.pkl')

    earlystop = EarlyStopping(patience=args.Optim.patience,
                              save_path=best_model_path,
                              descend_mode=True)

    # wandb init
    wandb_name = reduce(lambda x, y: x+'+'+y, args.Data.features) + '_' + args.Model.model + str(i)
    wandb_name = wandb_name + '_' + os.uname().nodename

    args.num_paras = get_param_num(model)

    try:
        import wandb
        wandb.init(project='Intention_recognition', name=wandb_name, dir=his_path, config=args)
    except ImportError:
        print

    # save json and para
    args.dump(his_path, wandb_name+'.json')

    with open(os.path.join(his_path,'paras.txt'),'w') as f:
        for argid, value in args.to_dict().items():
            f.writelines(argid + ':' + str(value) + '\n')

    # training
    lossfunc = Multi_weighted_crossentropyloss(args.Data.labels, weighted=args.Loss.weighted_loss, label_weight=args.Loss.label_weight, device=device)

    train_eval = Train_Eval_Model(model,
                                load_model_path=args.Model.load_model_path,
                                optim=args.Optim.optimizer,
                                lr=args.Optim.lr,
                                loss_func=lossfunc,
                                device=device,
                                lr_scheduler=args.Optim.lr_scheduler,
                                scheduler_type=args.Optim.scheduler_type,
                                scheduler_paras=args.Optim.scheduler_paras)

    _, _, _, _, _, _, labels, preds = train_eval.train_model(
                                train_loader,
                                val_loader,
                                test_loader,
                                epoch_num=args.Optim.epoch_num,
                                earlystop=earlystop)

    if os.path.exists('results/result.pkl'):
        with open('results/result.pkl','rb') as f:
            results = pickle.load(f)
    else:
        results = {}

    def result_ana(labels, preds):
        result = {}
        result['accuracy'] = accuracy_score(labels, preds) * 100
        result['fscore'] = f1_score(labels, preds, average='macro') * 100
        result['recall'] = recall_score(labels, preds, average='macro') * 100
        result['labels'] = labels
        result['preds'] = preds
        return result

    result = result_ana(labels, preds)
    print(classification_report(labels, preds))
    
    with open(os.path.join(his_path, 'result.pkl'),'wb') as f:
        pickle.dump(result, f)

    return earlystop.save_path, result

if __name__ == '__main__':

    args = JsonConfig('hparas/gcn_warmup.json')

    print(args)

    main(args)