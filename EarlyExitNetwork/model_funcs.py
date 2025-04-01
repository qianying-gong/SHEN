# model_funcs.py
# implements the functions for training, testing SDNs and CNNs
# also implements the functions for computing confusion and confidence

import torch
import time
import torch.nn as nn
import numpy as np
import aux_funcs as af
import data


def sdn_training_step(optimizer, model, coeffs, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]
        cur_loss = float(coeffs[ic_id])*af.get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss += af.get_loss_criterion()(output[-1], b_y)
    total_loss.backward()
    optimizer.step()                # apply gradients

    return total_loss

def sdn_ic_only_step(optimizer, model, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    for output_id, cur_output in enumerate(output):
        if output_id == model.num_output - 1: # last output
            break
        
        cur_loss = af.get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss.backward()
    optimizer.step()                # apply gradients

    return total_loss

def get_loader(data, augment):
    if augment:
        train_loader = data.aug_train_loader
    else:
        train_loader = data.train_loader

    return train_loader  


def sdn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
    augment = model.augment_training
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}
    add_output = model.add_output
    macs = [0.0415, 0.07916, 0.11682, 0.15448, 0.19214, 0.2298, 0.26746, 0.30512, 0.34278, 0.37198, 0.40925, 0.44652,
            0.48379, 0.52106, 0.55833, 0.5956, 0.63287, 0.67014, 0.69909, 0.73617, 0.77325, 0.81033, 0.84741, 0.88449,
            0.92157, 0.95865, 0.99573]
    max_coeffs = np.array([macs[i] for i in range(len(add_output)) if add_output[i] == 1])

    for epoch in range(1, epochs+1):
        scheduler.step()
        cur_lr = af.get_lr(optimizer)
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        if model.ic_only is False:
            # calculate the IC coeffs for this epoch for the weighted objective function
            cur_coeffs = 0.01 + epoch*(max_coeffs/epochs) # to calculate the tau at the currect epoch
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)

        model.train()

        top1_test, top5_test = sdn_test(model, data.test_loader, device)

        print('Top1 Test accuracies: {}'.format(top1_test))

        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
        print('Top1 Train accuracies: {}'.format(top1_train))
        # print('Top5 Train accuracies: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)

        metrics['lrs'].append(cur_lr)

    return metrics

def sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = []
    top5 = []
    for output_id in range(model.num_output):
        t1 = data.AverageMeter()
        t5 = data.AverageMeter()
        top1.append(t1)
        top5.append(t5)

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec5 = data.accuracy(cur_output, b_y, topk=(1, 5))
                top1[output_id].update(prec1[0], b_x.size(0))
                top5[output_id].update(prec5[0], b_x.size(0))


    top1_accs = []
    top5_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
        top5_accs.append(top5[output_id].avg.data.cpu().numpy()[()])

    return top1_accs, top5_accs

def sdn_get_detailed_results(model, loader, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            for output_id in outputs:
                cur_output = output[output_id]
                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

                pred = cur_output.max(1, keepdim=True)[1]
                is_correct = pred.eq(b_y.view_as(pred))
                for test_id in range(len(b_x)):
                    cur_instance_id = test_id + cur_batch_id*loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, layer_predictions, layer_confidence


def sdn_get_confusion(model, loader, confusion_stats, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    instance_confusion = {}
    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, confusion_stats, device)
            
            for test_id in range(len(b_x)):
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
                instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
                for output_id in outputs:
                    cur_output = output[output_id]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    correct = is_correct[test_id]
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, instance_confusion

# to normalize the confusion scores
def sdn_confusion_stats(model, loader, device='cpu'):
    model.eval()
    outputs = list(range(model.num_output))
    confusion_scores = []

    total_num_instances = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            total_num_instances += len(b_x)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, None, device)
            for test_id in range(len(b_x)):
                confusion_scores.append(cur_confusion[test_id].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    mean_con = float(np.mean(confusion_scores))
    std_con = float(np.std(confusion_scores))
    return (mean_con, std_con)

def sdn_test_early_exits(model, loader, device='cpu'):
    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output

    top1 = data.AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)

            output, output_id, is_early = model(b_x)

            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

            prec1 = data.accuracy(output, b_y, topk=(1,5))
            top1.update(prec1[0], b_x.size(0))
    top1_acc = top1.avg.data.cpu().numpy()[()]

    return top1_acc, early_output_counts, non_conf_output_counts