# train_networks.py
# For training CNNs and SDNs via IC-only and SDN-training strategies
# It trains and save the resulting models to an output directory specified in the main function


import aux_funcs  as af
import network_architectures as arcs

from architectures.CNNs.VGG import VGG
# def train(models_path, untrained_models, sdn=False, ic_only_sdn=False, device='cuda:1'):
def train(models_path, base_model, sdn=False, ic_only_sdn=False, device='cuda:1'):
    # print('Training models...')

    # for base_model in untrained_models:
    trained_model, model_params = arcs.load_model(models_path, base_model, 0)
    dataset = af.get_dataset(model_params['task'])

    learning_rate = model_params['learning_rate']
    momentum = model_params['momentum']
    weight_decay = model_params['weight_decay']
    milestones = model_params['milestones']
    gammas = model_params['gammas']
    num_epochs = model_params['epochs']

    model_params['optimizer'] = 'SGD'

    if ic_only_sdn:  # IC-only training, freeze the original weights
        learning_rate = model_params['ic_only']['learning_rate']
        num_epochs = model_params['ic_only']['epochs']
        milestones = model_params['ic_only']['milestones']
        gammas = model_params['ic_only']['gammas']

        model_params['optimizer'] = 'Adam'

        trained_model.ic_only = True


    optimization_params = (learning_rate, weight_decay, momentum)
    lr_schedule_params = (milestones, gammas)

    if sdn:
        if ic_only_sdn:
            optimizer, scheduler = af.get_sdn_ic_only_optimizer(trained_model, optimization_params, lr_schedule_params)
            trained_model_name = base_model+'_ic_only'

        else:
            optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
            trained_model_name = base_model+'_sdn_training'

    else:
            optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
            trained_model_name = base_model

    print(trained_model_name)
    trained_model.to(device)
    metrics = trained_model.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device=device)
    model_params['train_top1_acc'] = metrics['train_top1_acc']
    model_params['test_top1_acc'] = metrics['test_top1_acc']
    model_params['train_top5_acc'] = metrics['train_top5_acc']
    model_params['test_top5_acc'] = metrics['test_top5_acc']
    model_params['epoch_times'] = metrics['epoch_times']
    model_params['lrs'] = metrics['lrs']
    total_training_time = sum(model_params['epoch_times'])
    model_params['total_time'] = total_training_time
    # print('Training took {} seconds...'.format(total_training_time))
    arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

def train_sdns(models_path, sdn_name, pos_exits, ic_only=False, device='cuda:1'):
    if ic_only: # if we only train the ICs, we load a pre-trained CNN
        load_epoch = -1
    else:  # if we train both ICs and the orig network, we load an untrained CNN
        load_epoch = 0

    # for sdn_name in networks:
    cnn_to_tune = sdn_name.replace('sdn', 'cnn')
    sdn_params = arcs.load_params(models_path, sdn_name)
    sdn_params = arcs.get_net_params(sdn_params['network_type'], sdn_params['task'], pos_exits)
    sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params, load_epoch)  # load the CNN and convert it to a SDN
    arcs.save_model(sdn_model, sdn_params, models_path, sdn_name, epoch=0)  # save the resulting SDN
    train(models_path, sdn_name, sdn=True, ic_only_sdn=ic_only, device=device)


def train_models(dataset, model, models_path, pos_exits, device='cpu'):
    # tasks = ['cifar10', 'cifar100', 'tinyimagenet']
    task = dataset

    if model == 'resnet56':
        cnn_name, sdn = arcs.create_resnet56(models_path, task, pos_exits, save_type='cd')  # save_type='cd'
    elif model == 'vgg16bn':
        cnn_name, sdn = arcs.create_vgg16bn(models_path, task, pos_exits, save_type='cd')
    elif model == 'mobilenet':
        cnn_name, sdn = arcs.create_mobilenet(models_path, task, pos_exits, save_type='cd')

    train_sdns(models_path, sdn, pos_exits, ic_only=False, device=device)  # train SDNs with SDN-training strategy

