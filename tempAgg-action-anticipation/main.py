# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from dataset import SequenceDataset
from os.path import join
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.utils import ValueMeter, topk_accuracy, topk_recall, MeanTopKRecallMeter
from utils.utils import get_marginal_indexes, marginalize, softmax, predictions_to_json
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from network import Network
from torch.optim import lr_scheduler
from torch import nn
import copy

view_list = ['view1', 'view2', 'view3', 'view4', 'view5', 'view6', 'view7', 'view8', 
                'view1+view2+view3+view4', 'view1+view2+view3+view4+view5+view6+view7+view8', 'all']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pd.options.display.float_format = '{:05.2f}'.format

parser = ArgumentParser(description="Training for Action Anticipation")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate', 'test'],
                    help="Whether to perform training, validation or test. If test/validate_json is selected, "
                         "--save_json must be used to provide a directory in which to save the generated jsons.")
parser.add_argument('--trainval', action='store_true', help='Whether to train on trainval.csv or only train.csv')
parser.add_argument('--path_to_data', type=str, default='data/', help="Path to the data folder, containing the .lmdb dataset")
parser.add_argument('--path_to_anno', type=str, default='data/', help="Path to the annotations folder")
parser.add_argument('--path_to_models', type=str, default='checkpoints', 
                    help="Path to the directory where all the models will be saved and loaded from")
parser.add_argument('--add_suffix', type=str, default=None, help="special suffix to add at the end of the model name")

parser.add_argument('--task', type=str, default='anticipation', choices=['anticipation'], help='Task is anticipation.')

parser.add_argument('--img_tmpl', type=str, default='{:010d}.jpg',
                    help='Template to use to load the representation of a given frame')
parser.add_argument('--resume', action='store_true', help='Whether to resume suspended training')
parser.add_argument('--best_model', type=str, default='best', choices=['best', 'last'], help='')

parser.add_argument('--modality', type=str, default='fixed', choices=['fixed', 'ego', 'fixed+ego'],
                    help="Using fixed or egocentric video stream or both.")
parser.add_argument('--views', type=str, default='view1', choices=view_list, help="which views to use")

parser.add_argument('--num_workers', type=int, default=0, help="Number of parallel thread to fetch the data")
parser.add_argument('--display_every', type=int, default=10, help="Display every n iterations")

parser.add_argument('--schedule_on', type=int, default=1, help='')
parser.add_argument('--schedule_epoch', type=int, default=10, help='')

parser.add_argument('--action_class', type=int, default=1064)
parser.add_argument('--verb_class', type=int, default=17)
parser.add_argument('--object_class', type=int, default=90)
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--linear_dim', type=int, default=512)
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--scale_factor', type=float, default=-.5)
parser.add_argument('--scale', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--epochs', type=int, default=15, help="Training epochs")
parser.add_argument('--video_feat_dim', type=int, default=2048, choices=[1024, 2048], help='dimension of the per-frame feature')
parser.add_argument('--past_attention', action='store_true')

# Spanning snippets
parser.add_argument('--spanning_sec', type=float, default=6, help='')
parser.add_argument('--span_dim1', type=int, default=5, help='')
parser.add_argument('--span_dim2', type=int, default=3, help='')
parser.add_argument('--span_dim3', type=int, default=2, help='')

# Recent snippets
parser.add_argument('--recent_dim', type=int, default=2, help='')
parser.add_argument('--recent_sec1', type=float, default=1.6, help='')
parser.add_argument('--recent_sec2', type=float, default=1.2, help='')
parser.add_argument('--recent_sec3', type=float, default=0.8, help='')
parser.add_argument('--recent_sec4', type=float, default=0.4, help='')

# Adding verb and object loss
parser.add_argument('--verb_object_scores', type=bool, default=True, help='')
parser.add_argument('--add_verb_loss', type=bool, default=True, help='Whether to train with verb loss or not')
parser.add_argument('--add_object_loss', type=bool, default=True, help='Whether to train with verb loss or not')
parser.add_argument('--verb_loss_weight', type=float, default=1.0, help='')
parser.add_argument('--object_loss_weight', type=float, default=1.0, help='')

parser.add_argument('--topK', type=int, default=1, help='')

# during evaluation (validation/testing)
parser.add_argument('--save_json', type=str, default=None, help='Directory in which to save the generated .json')
parser.add_argument('--model_name_ev', type=str, default=None, help="name of the pretrained model for evaluating")

parser.add_argument('--alpha', type=float, default=1, help="Distance between time-steps in seconds")

# Debugging True
parser.add_argument('--debug_on', type=bool, default=False, help='')


args = parser.parse_args()

view_dict = {'fixed': {'view1': 'C10095_rgb',
                    'view2' : 'C10115_rgb',
                    'view3' : 'C10118_rgb',
                    'view4' : 'C10119_rgb',
                    'view5' : 'C10379_rgb',
                    'view6' : 'C10390_rgb',
                    'view7' : 'C10395_rgb',
                    'view8' : 'C10404_rgb'},
            'ego': {'view1': ['HMC_21176875_mono10bit', 'HMC_84346135_mono10bit'],
                    'view2' : ['HMC_21176623_mono10bit', 'HMC_84347414_mono10bit'],
                    'view3' : ['HMC_21110305_mono10bit', 'HMC_84355350_mono10bit'],
                    'view4' : ['HMC_21179183_mono10bit', 'HMC_84358933_mono10bit']}}

view = args.views
if '+' in args.modality:
    modality = args.modality.split('+')
    args.modality = 'combined'
else:
    modality = [args.modality]

out_views = []
for modal in modality:
    if modal == 'fixed':
        if args.views == 'all':
            vs = ('+').join(list(view_dict[modal].keys()))
        out_views.extend([view_dict[modal][x] for x in vs.strip(' ').split('+')])
    elif modal == 'ego':
        if args.views == 'all':
            vs = list(view_dict[modal].keys())
        else:
            vs = args.views.strip(' ').split('+')
        out_v = []
        for x in vs:
            out_v.extend(view_dict[modal][x])
        out_views.extend(out_v)
args.views = out_views.copy()
print(f'Modality = {args.modality}')
print(f'Views being used for "{args.mode}" = {args.views}')
if args.mode == 'train':
    print(f'[trainval] {args.trainval}')

def make_model_name(arg_save):
    global view
    save_name = "anti_mod_{}_{}_span_{}_s1_{}_s2_{}_s3_{}_recent_{}_r1_{}_r2_{}_r3_{}_r4_{}_bs_{}_drop_{}_lr_{}_dimLa_{}_" \
                "dimLi_{}_epoc_{}".format(arg_save.modality, view, arg_save.spanning_sec, arg_save.span_dim1,
                                          arg_save.span_dim2, arg_save.span_dim3, arg_save.recent_dim,
                                          arg_save.recent_sec1, arg_save.recent_sec2, arg_save.recent_sec3,
                                          arg_save.recent_sec4, arg_save.batch_size, arg_save.dropout_rate, arg_save.lr,
                                          arg_save.latent_dim, arg_save.linear_dim, arg_save.epochs)
    if arg_save.add_verb_loss:
        save_name = save_name + '_vb'
    if arg_save.add_object_loss:
        save_name = save_name + '_nn'
    if not args.add_suffix==None:
        save_name = f'{save_name}_{args.add_suffix}'

    return save_name


def save_model(model, optimizer, epoch, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'optimizer': optimizer.state_dict(), 'perf': perf, 'best_perf': best_perf}, join(
                args.path_to_models, exp_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 
                    'optimizer': optimizer.state_dict(), 'perf': perf, 'best_perf': best_perf}, join(
                    args.path_to_models, exp_name + '_best.pth.tar'))


def get_scores(model, loader, challenge=False, include_discarded=False):
    model.eval()
    predictions_act = []
    predictions_object = []
    predictions_verb = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x_spanning = batch['spanning_features']
            x_recent = batch['recent_features']
            
            if type(x_spanning) == list:
                x_spanning = [xx.to(device) for xx in x_spanning]
                x_recent = [xx.to(device) for xx in x_recent]
            else:
                x_spanning = x_spanning.to(device)
                x_recent = x_recent.to(device)

            y_label = batch['label'].numpy()
            ids.append(batch['id'])

            pred_act1, pred_act2, pred_act3, pred_act4, pred_verb1, pred_verb2, pred_verb3, pred_verb4, \
            pred_object1, pred_object2, pred_object3, pred_object4 = model(x_spanning, x_recent)

            pred_ensemble_act = pred_act1.detach() + pred_act2.detach() + pred_act3.detach() + pred_act4.detach()
            pred_ensemble_act = pred_ensemble_act.cpu().numpy()
            predictions_act.append(pred_ensemble_act)
            
            if args.verb_object_scores:
                pred_ensemble_verb = pred_verb1.detach() + pred_verb2.detach() + pred_verb3.detach() + pred_verb4.detach()
                pred_ensemble_verb = pred_ensemble_verb.cpu().numpy()
                pred_ensemble_object = pred_object1.detach() + pred_object2.detach() + pred_object3.detach() + pred_object4.detach()
                pred_ensemble_object = pred_ensemble_object.cpu().numpy()

                predictions_verb.append(pred_ensemble_verb)
                predictions_object.append(pred_ensemble_object)
            
            labels.append(y_label)

    action_scores = np.concatenate(predictions_act)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    if args.verb_object_scores:  # use the verb and object scores
        verb_scores = np.concatenate(predictions_verb)
        object_scores = np.concatenate(predictions_object)
    else:  # marginalize the action scores to get the object and verb scores
        actions = pd.read_csv(join(args.path_to_data, 'actions.csv'), index_col='id')
        vi = get_marginal_indexes(actions, 'verb')
        ni = get_marginal_indexes(actions, 'object')
        action_prob = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
        verb_scores = marginalize(action_prob, vi) 
        object_scores = marginalize(action_prob, ni)

    if challenge or include_discarded:
        dlab = np.array(loader.dataset.discarded_labels)
        dislab = np.array(loader.dataset.discarded_ids)
        ids = np.concatenate([ids, dislab])
        num_disc = len(dlab)
        labels = np.concatenate([labels, dlab])
        verb_scores = np.concatenate((verb_scores, np.zeros((num_disc, *verb_scores.shape[1:]))))
        object_scores = np.concatenate((object_scores, np.zeros((num_disc, *object_scores.shape[1:]))))
        action_scores = np.concatenate((action_scores, np.zeros((num_disc, *action_scores.shape[1:]))))

    if labels.max() > 0 and not challenge:
        return verb_scores, object_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2], ids
    else:
        return verb_scores, object_scores, action_scores, ids



def fuse_per_view_scores(ids, action_scores, verb_scores, object_scores, fusion='avgpool'):
    print('Fusing per view scores...')

    scores = dict()
    ids = ids.astype(int)
    
    for i in range(len(ids)):
        if not ids[i] in scores:
            scores[ids[i]] = dict()
            scores[ids[i]]['action'] = []
            scores[ids[i]]['verb'] = []
            scores[ids[i]]['object'] = []

        scores[ids[i]]['action'].append(action_scores[i])
        scores[ids[i]]['verb'].append(verb_scores[i])
        scores[ids[i]]['object'].append(object_scores[i])

    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[0])}
    ids = np.array(list(scores.keys()))
    
    action_scores = [np.array(scores[ids]['action']) for ids in scores]
    verb_scores = [np.array(scores[ids]['verb']) for ids in scores]
    object_scores = [np.array(scores[ids]['object']) for ids in scores]

    if fusion == 'avgpool':
        action_scores = np.array([np.mean(x, axis=0) for x in action_scores])
        verb_scores = np.array([np.mean(x, axis=0) for x in verb_scores])
        object_scores = np.array([np.mean(x, axis=0) for x in object_scores])
    elif fusion == 'maxpool':
        action_scores = np.array([np.max(x, axis=0) for x in action_scores])
        verb_scores = np.array([np.max(x, axis=0) for x in verb_scores])
        object_scores = np.array([np.max(x, axis=0) for x in object_scores])
    else:
        raise NotImplementedError(f'Fusion scheme {fusion} not implemented.')
    
    return ids, action_scores, verb_scores, object_scores



def log(mode, epoch, total_loss_meter, ensemble_accuracy_meter,
        action_loss_meter, verb_loss_meter, object_loss_meter,
        accuracy_action1_meter, accuracy_action2_meter, accuracy_action3_meter, accuracy_action4_meter,
        best_perf=None, green=False):
    if green:
        print('\033[92m', end="")
    print(
        "[{}] Epoch: {:.2f}. ".format(mode, epoch),
        "Total Loss: {:.2f}. ".format(total_loss_meter.value()),
        "Act. Loss: {:.2f}. ".format(action_loss_meter.value()),
        "Verb Loss: {:.2f}. ".format(verb_loss_meter.value()),
        "Object Loss: {:.2f}. ".format(object_loss_meter.value()),
        "Acc. Act1: {:.2f}% ".format(accuracy_action1_meter.value()),
        "Acc. Act2: {:.2f}% ".format(accuracy_action2_meter.value()),
        "Acc. Act3: {:.2f}% ".format(accuracy_action3_meter.value()),
        "Acc. Act4: {:.2f}% ".format(accuracy_action4_meter.value()),
        "Ensemble Acc.: {:.2f}% ".format(ensemble_accuracy_meter.value()),
        end="")

    if best_perf:
        print("[best: {:.2f}]%".format(best_perf), end="")

    print('\033[0m')


def train_validation(model, loaders, optimizer, epochs, start_epoch, start_best_perf, schedule_on):
    """Training/Validation code"""

    best_perf = start_best_perf  # to keep track of the best performing epoch

    loss_act_TAB1 = nn.CrossEntropyLoss()
    loss_act_TAB2 = nn.CrossEntropyLoss()
    loss_act_TAB3 = nn.CrossEntropyLoss()
    loss_act_TAB4 = nn.CrossEntropyLoss()
    if args.add_verb_loss:
        print('Add verb losses')
        loss_verb_TAB1 = nn.CrossEntropyLoss()
        loss_verb_TAB2 = nn.CrossEntropyLoss()
        loss_verb_TAB3 = nn.CrossEntropyLoss()
        loss_verb_TAB4 = nn.CrossEntropyLoss()
    if args.add_object_loss:
        print('Add object losses')
        loss_object_TAB1 = nn.CrossEntropyLoss()
        loss_object_TAB2 = nn.CrossEntropyLoss()
        loss_object_TAB3 = nn.CrossEntropyLoss()
        loss_object_TAB4 = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epochs):
        if schedule_on is not None:
            schedule_on.step()

        # define training and validation meters
        total_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        action_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        verb_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        object_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}

        ensemble_accuracy_meter = {'training': MeanTopKRecallMeter(args.action_class), 'validation': MeanTopKRecallMeter(args.action_class)}
        accuracy_action1_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_action2_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_action3_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_action4_meter = {'training': ValueMeter(), 'validation': ValueMeter()}

        for mode in ['training', 'validation']:

            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):
                    x_spanning = batch['spanning_features']
                    x_recent = batch['recent_features']
                    if type(x_spanning) == list:
                        x_spanning = [xx.to(device) for xx in x_spanning]
                        x_recent = [xx.to(device) for xx in x_recent]
                    else:
                        x_spanning = x_spanning.to(device)
                        x_recent = x_recent.to(device)

                    y_label = batch['label'].to(device)
                    bs = y_label.shape[0]  # batch size

                    pred_act1, pred_act2, pred_act3, pred_act4, pred_verb1, pred_verb2, pred_verb3, pred_verb4, \
                    pred_object1, pred_object2, pred_object3, pred_object4 = model(x_spanning, x_recent)
                    
                    loss = loss_act_TAB1(pred_act1, y_label[:, 2]) + \
                           loss_act_TAB2(pred_act2, y_label[:, 2]) + \
                           loss_act_TAB3(pred_act3, y_label[:, 2]) + \
                           loss_act_TAB4(pred_act4, y_label[:, 2])
                    action_loss_meter[mode].add(loss.item(), bs)

                    if args.add_verb_loss:
                        verb_loss = loss_verb_TAB1(pred_verb1, y_label[:, 0]) + \
                                    loss_verb_TAB2(pred_verb2, y_label[:, 0]) + \
                                    loss_verb_TAB3(pred_verb3, y_label[:, 0]) + \
                                    loss_verb_TAB4(pred_verb4, y_label[:, 0])
                        verb_loss_meter[mode].add(verb_loss.item(), bs)
                        loss = loss + args.verb_loss_weight * verb_loss
                    else:
                        verb_loss_meter[mode].add(-1, bs)

                    if args.add_object_loss:
                        object_loss = loss_object_TAB1(pred_object1, y_label[:, 1]) + \
                                    loss_object_TAB2(pred_object2, y_label[:, 1]) + \
                                    loss_object_TAB3(pred_object3, y_label[:, 1]) + \
                                    loss_object_TAB4(pred_object4, y_label[:, 1])
                        object_loss_meter[mode].add(object_loss.item(), bs)
                        loss = loss + args.object_loss_weight * object_loss
                    else:
                        object_loss_meter[mode].add(-1, bs)

                    label_curr = y_label[:, 2].detach().cpu().numpy()
                    acc_future1 = topk_accuracy(pred_act1.detach().cpu().numpy(), label_curr, (args.topK,))[0] * 100
                    acc_future2 = topk_accuracy(pred_act2.detach().cpu().numpy(), label_curr, (args.topK,))[0] * 100
                    acc_future3 = topk_accuracy(pred_act3.detach().cpu().numpy(), label_curr, (args.topK,))[0] * 100
                    acc_future4 = topk_accuracy(pred_act4.detach().cpu().numpy(), label_curr, (args.topK,))[0] * 100
                    
                    accuracy_action1_meter[mode].add(acc_future1, bs)
                    accuracy_action2_meter[mode].add(acc_future2, bs)
                    accuracy_action3_meter[mode].add(acc_future3, bs)
                    accuracy_action4_meter[mode].add(acc_future4, bs)

                    pred_ensemble = pred_act1.detach() + pred_act2.detach() + pred_act3.detach() + pred_act4.detach()
                    pred_ensemble = pred_ensemble.cpu().numpy()
                    
                    # store the values in the meters to keep incremental averages
                    total_loss_meter[mode].add(loss.item(), bs)
                    ensemble_accuracy_meter[mode].add(pred_ensemble, label_curr)

                    # if in training mode
                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # log training during loop - avoid logging the very first batch. It can be biased.
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        epoch_curr = epoch + i / len(loaders[mode])  # compute decimal epoch for logging
                        log(mode, epoch_curr, total_loss_meter[mode], ensemble_accuracy_meter[mode],
                            action_loss_meter[mode], verb_loss_meter[mode], object_loss_meter[mode],
                            accuracy_action1_meter[mode], accuracy_action2_meter[mode],
                            accuracy_action3_meter[mode], accuracy_action4_meter[mode])

                # log at the end of each epoch
                log(mode, epoch + 1, total_loss_meter[mode], ensemble_accuracy_meter[mode],
                    action_loss_meter[mode], verb_loss_meter[mode], object_loss_meter[mode],
                    accuracy_action1_meter[mode], accuracy_action2_meter[mode],
                    accuracy_action3_meter[mode], accuracy_action4_meter[mode],
                    max(ensemble_accuracy_meter[mode].value(), best_perf) if mode == 'validation' else None, green=True)

        if best_perf < ensemble_accuracy_meter['validation'].value():
            best_perf = ensemble_accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False
        with open(args.path_to_models + '/' + exp_name + '.txt', 'a') as f:
            f.write("%d - %0.2f\n" % (epoch + 1, ensemble_accuracy_meter['validation'].value()))

        # save checkpoint at the end of each train/val epoch
        save_model(model, optimizer, epoch + 1, ensemble_accuracy_meter['validation'].value(), best_perf, is_best=is_best)

    with open(args.path_to_models + '/' + exp_name + '.txt', 'a') as f:
        f.write("%d - %0.2f\n" % (epochs + 1, best_perf))


def load_checkpoint(model, optimizer=None):
    model_add = '.pth.tar'
    if args.best_model == 'best':
        print('args.best_model == True')
        model_add = '_best.pth.tar'

    chk = torch.load(join(args.path_to_models, exp_name + model_add))
    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']
    sd = {k.replace('noun', 'object') : v for k,v in chk['state_dict'].items()}

    model.load_state_dict(sd)

    if optimizer:
        optimizer.load_state_dict(chk['optimizer'])
    
    return epoch, perf, best_perf


def get_loader(mode, override_modality=None):
    if override_modality:
        path_to_lmdb = join(args.path_to_data, override_modality)
        img_template = args.img_tmpl
    else:
        path_to_lmdb = args.path_to_data
        img_template = args.img_tmpl

    if mode=='training' and args.trainval:
        csv_file = 'trainval'
    else:
        csv_file = mode

    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_anno, f'{csv_file}.csv'),
        'time_step': args.alpha,
        'label_type': ['verb', 'object', 'action'],
        'img_tmpl': img_template,
        'challenge': 'test' in mode,
        'args': args
    }
    _set = SequenceDataset(**kargs)

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, drop_last=mode == 'training', shuffle=mode == 'training')


def get_model():
    return Network(args)

def main():
    model = get_model()
    if type(model) == list:
        model = [m.to(device) for m in model]
    else:
        model.to(device)

    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.resume:
            start_epoch, _, start_best_perf = load_checkpoint(model, optimizer)

        else:
            start_epoch = 0
            start_best_perf = 0

        schedule_on = None
        if args.schedule_on:
            schedule_on = lr_scheduler.StepLR(optimizer, args.schedule_epoch, gamma=0.1, last_epoch=-1)

        train_validation(model, loaders, optimizer, args.epochs, 
                            start_epoch, start_best_perf, schedule_on)

    elif args.mode == 'validate':
        if args.model_name_ev == None:
            epoch, perf, _ = load_checkpoint(model)
            print(f'Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:.2f}.')
        else:
            chk = torch.load(f'{args.path_to_models}/{args.model_name_ev}')
            epoch = chk['epoch']
            best_perf = chk['best_perf']
            perf = chk['perf']
            model.load_state_dict(chk['state_dict'])
            print(f'Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:.2f}.')

        loader = get_loader('validation')
        verb_scores, object_scores, action_scores, verb_labels, object_labels, action_labels, ids = get_scores(model, loader, include_discarded=(args.save_json!=None))
        
        verb_accuracies = topk_accuracy(verb_scores, verb_labels, (1,))[0]
        object_accuracies = topk_accuracy(object_scores, object_labels, (1,))[0]
        action_accuracies = topk_accuracy(action_scores, action_labels, (1,))[0]

        verb_accuracies_5 = topk_accuracy(verb_scores, verb_labels, (5,))[0]
        object_accuracies_5 = topk_accuracy(object_scores, object_labels, (5,))[0]
        action_accuracies_5 = topk_accuracy(action_scores, action_labels, (5,))[0]

        verb_recalls = topk_recall(verb_scores, verb_labels, k=1)
        object_recalls = topk_recall(object_scores, object_labels, k=1)
        action_recalls = topk_recall(action_scores, action_labels, k=1)

        verb_recalls_5 = topk_recall(verb_scores, verb_labels, k=5)
        object_recalls_5 = topk_recall(object_scores, object_labels, k=5)
        action_recalls_5 = topk_recall(action_scores, action_labels, k=5)

        print(f'Overall Top-1 Acc. (Verb) = {100*topk_accuracy(verb_scores, verb_labels, (1,))[0]:.2f}')
        print(f'Overall Top-1 Acc. (Object) = {100*topk_accuracy(object_scores, object_labels, (1,))[0]:.2f}')
        print(f'Overall Top-1 Acc. (Action) = {100*topk_accuracy(action_scores, action_labels, (1,))[0]:.2f}')
        print(f'Overall Top-5 Acc. (Verb) = {100*topk_accuracy(verb_scores, verb_labels, (5,))[0]:.2f}')
        print(f'Overall Top-5 Acc. (Object) = {100*topk_accuracy(object_scores, object_labels, (5,))[0]:.2f}')
        print(f'Overall Top-5 Acc. (Action) = {100*topk_accuracy(action_scores, action_labels, (5,))[0]:.2f}')

        print(f'Overall Top-1 Recall (Verb) = {100*topk_recall(verb_scores, verb_labels, k=1):.2f}')
        print(f'Overall Top-1 Recall (Object) = {100*topk_recall(object_scores, object_labels, k=1):.2f}')
        print(f'Overall Top-1 Recall (Action) = {100*topk_recall(action_scores, action_labels, k=1):.2f}')
        print(f'Overall Top-5 Recall (Verb) = {100*topk_recall(verb_scores, verb_labels, k=5):.2f}')
        print(f'Overall Top-5 Recall (Object) = {100*topk_recall(object_scores, object_labels, k=5):.2f}')
        print(f'Overall Top-5 Recall (Action) = {100*topk_recall(action_scores, action_labels, k=5):.2f}')

        if args.save_json:
            ids, action_scores, verb_scores, object_scores = fuse_per_view_scores(ids, action_scores, verb_scores, object_scores)
            predictions = predictions_to_json(args.task, ids, action_scores, verb_scores, object_scores)

            with open(join(args.save_json, "preds.json"), 'w') as f:
                f.write(json.dumps(predictions, indent=4, separators=(',', ': ')))
            print(f'.json saved in {args.save_json}.')


    elif args.mode == 'test':
        if args.model_name_ev == None:
            epoch, perf, _ = load_checkpoint(model)
            print(f'Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:.2f}.')
        else:
            chk = torch.load(f'{args.path_to_models}/{args.model_name_ev}')
            epoch = chk['epoch']
            best_perf = chk['best_perf']
            perf = chk['perf']
            model.load_state_dict(chk['state_dict'])
            print(f'Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:.2f}.')

        loader = get_loader('test')
        
        verb_scores, object_scores, action_scores, ids = get_scores(model, loader, challenge=True)
        ids, action_scores, verb_scores, object_scores = fuse_per_view_scores(ids, action_scores, verb_scores, object_scores)

        predictions = predictions_to_json(args.task, ids, action_scores, verb_scores, object_scores)

        with open(join(args.save_json, "preds.json"), 'w') as f:
            f.write(json.dumps(predictions, indent=4, separators=(',', ': ')))
        print(f'.json saved in {args.save_json}.')



if __name__ == '__main__':

    if args.mode == 'test':
        assert args.save_json is not None

    exp_name = make_model_name(args)
    print("Save file name ", exp_name)
    print("Printing Arguments ")
    print(args)

    main()