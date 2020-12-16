import torch
import network
from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import lr_schedule
import copy
import os
import utils
import torch.nn.functional as F
from torch import autograd
import pdb

def image_classification_test(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def train(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(args.save_path).readlines(),
                                transform=image_train(),pseudo=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=4)

    #model
    model = network.ResNet(class_num=args.num_class,radius=args.radius).cuda()
    adv_net = network.AdversarialNetwork(in_feature=model.output_num(),hidden_size=1024,max_iter=2000).cuda()
    parameter_classifier = [model.get_parameters()[2]]
    parameter_feature = model.get_parameters()[0:2] + adv_net.get_parameters()
    optimizer_classifier = torch.optim.SGD(parameter_classifier,lr=args.lr,momentum=0.9,weight_decay=0.005)
    optimizer_feature = torch.optim.SGD(parameter_feature,lr=args.lr,momentum=0.9,weight_decay=0)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)

    Cs_memory = torch.zeros(args.num_class, 256).cuda()
    Ct_memory = torch.zeros(args.num_class, 256).cuda()

    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\n iter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model,'snapshot/save/best_model.pk')

        model.train(True)
        adv_net.train(True)
        if(args.lr_decay):
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier,i)
            optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, pseudo_labels_target, weights = iter_target.next()
        inputs_source, labels_source = inputs_source.cuda(),  labels_source.cuda()
        inputs_target, pseudo_labels_target = inputs_target.cuda(), pseudo_labels_target.cuda()
        weights = weights.type(torch.Tensor).cuda()

        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)

        source_class_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        adv_loss = utils.loss_adv(features,adv_net)
        H = torch.mean(utils.Entropy(F.softmax(outputs_target, dim=1)))
        target_robust_loss = utils.robust_pseudo_loss(outputs_target,pseudo_labels_target,weights)

        classifier_loss = source_class_loss + target_robust_loss
        optimizer_classifier.zero_grad()
        classifier_loss.backward(retain_graph=True)
        optimizer_classifier.step()

        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i,max_iter=2000)
        elif args.baseline =='DANN':
            lam = 0.0
        pseu_labels_target = torch.argmax(outputs_target, dim=1)
        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target,
                                                Cs_memory, Ct_memory)
        feature_loss = classifier_loss + adv_loss + lam*loss_sm + lam*H
        optimizer_feature.zero_grad()
        feature_loss.backward()
        optimizer_feature.step()

        print('step:{: d},\t,source_class_loss:{:.4f},\t,target_robust_loss:{:.4f}'
              ''.format(i, source_class_loss.item(),target_robust_loss.item()))

        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model


def penalty_loss_scale(loss, scale):
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def penalty_loss_scales(loss, scale):
    grads = autograd.grad(loss, scale, create_graph=True)
    #pdb.set_trace()
    return list(map(lambda x: torch.sum(x ** 2), grads))

def penalty(logits, y, criterion_fun):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = criterion_fun(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def penalty_weights(logits, y, weights, criterion_fun):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = criterion_fun(logits * scale, y, weights)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def train_irm_logit(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(args.save_path).readlines(),
                                transform=image_train(),pseudo=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=4)

    #model
    model = network.ResNet(class_num=args.num_class,radius=args.radius).cuda()
    parameter_classifier = model.get_parameters()
    optimizer_classifier = torch.optim.SGD(parameter_classifier,lr=args.lr,momentum=0.9,weight_decay=0.005)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)

    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\n iter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model,'snapshot/save/best_model.pk')

        model.train(True)
        if(args.lr_decay):
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier,i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, pseudo_labels_target, weights = iter_target.next()
        inputs_source, labels_source = inputs_source.cuda(),  labels_source.cuda()
        inputs_target, pseudo_labels_target = inputs_target.cuda(), pseudo_labels_target.cuda()
        weights = weights.type(torch.Tensor).cuda()

        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)

        source_class_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        #H = torch.mean(utils.Entropy(F.softmax(outputs_target, dim=1)))
        #pdb.set_trace()
        target_robust_loss = utils.robust_pseudo_loss(outputs_target,pseudo_labels_target,weights)

        source_irm_loss = penalty(outputs_source, labels_source, nn.CrossEntropyLoss())
        target_irm_loss = penalty(outputs_target, pseudo_labels_target, lambda x,y: utils.robust_pseudo_loss(x,y,weights))

        classifier_loss = source_class_loss + target_robust_loss
        classifier_loss += args.irm_weight * (source_irm_loss + target_irm_loss)
        #classifier_loss += args.irm_weight * (source_irm_loss)

        optimizer_classifier.zero_grad()
        #classifier_loss.backward()
        classifier_loss.backward()
        optimizer_classifier.step()

        # if args.baseline == 'MSTN':
        #     lam = network.calc_coeff(i,max_iter=2000)
        # elif args.baseline =='DANN':
        #     lam = 0.0
        # pseu_labels_target = torch.argmax(outputs_target, dim=1)
        # loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target,
        #                                         Cs_memory, Ct_memory)
        # feature_loss = classifier_loss + lam*loss_sm + lam*H
        # optimizer_feature.zero_grad()
        # feature_loss.backward()
        # optimizer_feature.step()

        print('step:{: d},\t,source_class_loss:{:.4f},\t,target_robust_loss:{:.4f}'
              ''.format(i, source_class_loss.item(),target_robust_loss.item()))

        #Cs_memory.detach_()
        #Ct_memory.detach_()

    return best_acc, best_model

def train_irm_feat(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(args.save_path).readlines(),
                                transform=image_train(),pseudo=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=4)

    #model
    model = network.ResNet(class_num=args.num_class,radius=args.radius).cuda()
    parameter_classifier = model.get_parameters()
    optimizer_classifier = torch.optim.SGD(parameter_classifier,lr=args.lr,momentum=0.9,weight_decay=0.005)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)

    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\n iter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model,'snapshot/save/best_model.pk')

        model.train(True)
        if(args.lr_decay):
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier,i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, pseudo_labels_target, weights = iter_target.next()
        inputs_source, labels_source = inputs_source.cuda(),  labels_source.cuda()
        inputs_target, pseudo_labels_target = inputs_target.cuda(), pseudo_labels_target.cuda()
        weights = weights.type(torch.Tensor).cuda()

        scale_source = torch.tensor(1.).cuda().requires_grad_()
        scale_target = torch.tensor(1.).cuda().requires_grad_()

        features_source, outputs_source = model.forward_mul(inputs_source, scale_source)
        features_target, outputs_target = model.forward_mul(inputs_target, scale_target)

        features = torch.cat((features_source, features_target), dim=0)

        source_class_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        target_robust_loss = utils.robust_pseudo_loss(outputs_target,pseudo_labels_target,weights)
        classifier_loss = source_class_loss + target_robust_loss

        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i,max_iter=2000)
        elif args.baseline =='DANN':
            lam = 0.0
        # pseu_labels_target = torch.argmax(outputs_target, dim=1)
        # loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target,
        #                                         Cs_memory, Ct_memory)
        # feature_loss = classifier_loss + lam*loss_sm + lam*H

        # irm_loss = 0
        # if('MSTN' in args.irm_feature):
        #     if(i>args.irm_warmup_step):
        #         irm_loss += sum(penalty_loss_scales(feature_loss, [scale_source, scale_target]))
        # else:
        source_irm_loss = penalty_loss_scale(source_class_loss, scale_source)
        target_irm_loss = penalty_loss_scale(target_robust_loss, scale_target)
        irm_loss = (source_irm_loss + target_irm_loss)
        
        feature_loss = classifier_loss + args.irm_weight * irm_loss

        optimizer_classifier.zero_grad()
        classifier_loss.backward()
        optimizer_classifier.step()

        print('step:{: d},\t,source_class_loss:{:.4f},\t,target_robust_loss:{:.4f}'
              ''.format(i, source_class_loss.item(),target_robust_loss.item()))

        #Cs_memory.detach_()
        #Ct_memory.detach_()

    return best_acc, best_model

def train_MSTN_irm_feat(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(args.save_path).readlines(),
                                transform=image_train(),pseudo=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=4)

    #model
    model = network.ResNet(class_num=args.num_class,radius=args.radius).cuda()
    parameter_classifier = model.get_parameters()
    optimizer_classifier = torch.optim.SGD(parameter_classifier,lr=args.lr,momentum=0.9,weight_decay=0.005)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)

    Cs_memory = torch.zeros(args.num_class, 256).cuda()
    Ct_memory = torch.zeros(args.num_class, 256).cuda()

    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\n iter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model,'snapshot/save/best_model.pk')

        model.train(True)
        if(args.lr_decay):
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier,i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, pseudo_labels_target, weights = iter_target.next()
        inputs_source, labels_source = inputs_source.cuda(),  labels_source.cuda()
        inputs_target, pseudo_labels_target = inputs_target.cuda(), pseudo_labels_target.cuda()
        weights = weights.type(torch.Tensor).cuda()

        if(args.irm_type == 'batch'):
            scale_source = torch.tensor(1.).cuda().requires_grad_()
            scale_target = torch.tensor(1.).cuda().requires_grad_()
        elif(args.irm_type == 'sample'):
            scale_source = torch.ones(inputs_source.size(0),1).cuda().requires_grad_()
            scale_target = torch.ones(inputs_target.size(0),1).cuda().requires_grad_()

        features_source, outputs_source = model.forward_mul(inputs_source, scale_source)
        features_target, outputs_target = model.forward_mul(inputs_target, scale_target)

        features = torch.cat((features_source, features_target), dim=0)

        source_class_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        target_robust_loss = utils.robust_pseudo_loss(outputs_target,pseudo_labels_target,weights)
        H = torch.mean(utils.Entropy(F.softmax(outputs_target, dim=1)))

        classifier_loss = source_class_loss + target_robust_loss

        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i,max_iter=2000)
        elif args.baseline =='DANN':
            lam = 0.0
        pseu_labels_target = torch.argmax(outputs_target, dim=1)
        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target,
                                                Cs_memory, Ct_memory)
        feature_loss = classifier_loss + lam*loss_sm + lam*H

        irm_loss = 0
        if(i > args.irm_warmup_step):
            irm_loss += sum(penalty_loss_scales(feature_loss, [scale_source, scale_target]))

        feature_loss += args.irm_weight * irm_loss

        optimizer_classifier.zero_grad()
        feature_loss.backward()
        optimizer_classifier.step()

        print('step:{: d},\t,source_class_loss:{:.4f},\t,target_robust_loss:{:.4f}'
              ''.format(i, source_class_loss.item(),target_robust_loss.item()))

        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model

def train_init(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(args.target_list).readlines(), \
                                transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=4)

    #model
    model = network.ResNet(class_num=args.num_class,radius=args.radius).cuda()
    adv_net = network.AdversarialNetwork(in_feature=model.output_num(),hidden_size=1024).cuda()
    parameter_list = model.get_parameters() + adv_net.get_parameters()
    optimizer = torch.optim.SGD(parameter_list,lr=args.lr,momentum=0.9,weight_decay=0.005)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)

    Cs_memory = torch.zeros(args.num_class, 256).cuda()
    Ct_memory = torch.zeros(args.num_class, 256).cuda()

    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\niter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model,'snapshot/save/initial_model.pk')

        model.train(True)
        adv_net.train(True)
        if(args.lr_decay):
            optimizer = lr_schedule.inv_lr_scheduler(optimizer,i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        adv_loss = utils.loss_adv(features,adv_net)

        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i)
        elif args.baseline =='DANN':
            lam = 0.0
        pseu_labels_target = torch.argmax(outputs_target, dim=1)
        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target,
                                                Cs_memory, Ct_memory)
        total_loss = classifier_loss + adv_loss + lam*loss_sm
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print('step:{: d},\t,class_loss:{:.4f},\t,adv_loss:{:.4f}'.format(i, classifier_loss.item(),
                                                                            adv_loss.item()))
        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model

def train_init_irm(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    dsets["source"] = ImageList(open(args.source_list).readlines(), \
                                transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(args.target_list).readlines(), \
                                transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(args.target_list).readlines(), \
                              transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=2 * args.batch_size, \
                                      shuffle=False, num_workers=4)

    #model
    model = network.ResNet(class_num=args.num_class,radius=args.radius).cuda()
    parameter_list = model.get_parameters()
    optimizer = torch.optim.SGD(parameter_list,lr=args.lr,momentum=0.9,weight_decay=0.005)

    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(model)

    Cs_memory = torch.zeros(args.num_class, 256).cuda()
    Ct_memory = torch.zeros(args.num_class, 256).cuda()

    for i in range(args.max_iter):
        if i % args.test_interval == args.test_interval - 1:
            model.train(False)
            temp_acc = image_classification_test(dset_loaders, model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(model)
            log_str = "\niter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model,'snapshot/save/initial_model.pk')

        model.train(True)
        if(args.lr_decay):
            optimizer = lr_schedule.inv_lr_scheduler(optimizer,i)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        if(args.irm_type == 'batch'):
            scale_source = torch.tensor(1.).cuda().requires_grad_()
            scale_target = torch.tensor(1.).cuda().requires_grad_()
        elif(args.irm_type == 'sample'):
            scale_source = torch.ones(inputs_source.size(0),1).cuda().requires_grad_()
            scale_target = torch.ones(inputs_target.size(0),1).cuda().requires_grad_()

        if(args.irm_feature == 'last_hidden'):
            features_source, outputs_source = model.forward_mul(inputs_source, scale_source)
            features_target, outputs_target = model.forward_mul(inputs_target, scale_target)
        elif(args.irm_feature == 'logit'):
            features_source, outputs_source = model(inputs_source)
            features_target, outputs_target = model(inputs_target)
            outputs_source = outputs_source * scale_source
            outputs_target = outputs_target * scale_target

        features = torch.cat((features_source, features_target), dim=0)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        
        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i)
        elif args.baseline =='DANN':
            lam = 0.0
        pseu_labels_target = torch.argmax(outputs_target, dim=1)

        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target,
                                                Cs_memory, Ct_memory)
        total_loss = classifier_loss + lam*loss_sm
       

        irm_loss = 0
        if(i>args.irm_warmup_step):
            if('MSTN' in args.init_method):
                irm_loss += sum(penalty_loss_scales(total_loss, [scale_source, scale_target]))
            else:
                source_irm_loss = sum(penalty_loss_scales(classifier_loss, [scale_source]))
                irm_loss += source_irm_loss        

            if('target' in args.init_method):
                classifier_loss_target = nn.CrossEntropyLoss()(outputs_target, pseu_labels_target.detach())
                target_irm_loss = sum(penalty_loss_scales(classifier_loss_target, [scale_target]))
                irm_loss += target_irm_loss
        
        total_loss += args.irm_weight * irm_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print('step:{: d},\t,class_loss:{:.4f},\t,irm_loss:{:.4f}'.format(i, classifier_loss.item(),
                                                                            float(irm_loss)))
        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model