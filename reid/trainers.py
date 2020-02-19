from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
import copy
import numpy as np
import os
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, model, AE_classifier, xi):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.AE_classifier = AE_classifier
        self.pid_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.xi = xi

    def train(self, epoch, data_loader, target_train_loader, optimizer, print_freq=1):
        self.set_model_train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        number_neighbor_epoch = torch.FloatTensor(0).zero_()
        end = time.time()

        if len(data_loader) > len(target_train_loader):
            long_data_loader = data_loader
            short_data_loader = target_train_loader
        else:
            long_data_loader = target_train_loader
            short_data_loader = data_loader

        # source iter
        short_iter = iter(short_data_loader)

        # Train
        for i, long_data in enumerate(long_data_loader):
            data_time.update(time.time() - end)

            try:
                short_data = next(short_iter)
            except:
                short_iter = iter(short_data_loader)
                short_data = next(short_iter)
            if len(data_loader) > len(target_train_loader):
                inputs, pids = self._parse_data(long_data)
                #for target data, pids are its indexes 
                #this is defined in data.py in the datasets folder
                inputs_target, index_target = self._parse_data(short_data)
            else:
                inputs, pids = self._parse_data(short_data)
                inputs_target, index_target = self._parse_data(long_data)

            # Source loss
            outputs = self.model(inputs)
            source_loss = self.pid_criterion(outputs, pids)
            prec, = accuracy(outputs.data, pids.data)
            prec1 = prec[0]

            # Target loss
            outputs = self.model(inputs_target, 'tgt_feat')
            target_loss, number_neighbor_iteration = self.AE_classifier(outputs, index_target, epoch=epoch)
            number_neighbor_epoch = torch.cat((number_neighbor_epoch, number_neighbor_iteration))
            loss = (1 - self.xi) * source_loss + self.xi * target_loss

            loss_print = {}
            loss_print['source_loss'] = source_loss.item()
            loss_print['target_loss'] = target_loss.item()

            losses.update(loss.item(), outputs.size(0))
            precisions.update(prec1, outputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}][{}/{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f}), Prec {:.2%} ({:.2%}) k_mean {:.3f} ({:.3f}) k_var {:.3f} ({:.3f})" \
                    .format(epoch, i + 1, len(long_data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg,
                            precisions.val, precisions.avg,
                            torch.mean(number_neighbor_iteration), torch.mean(number_neighbor_epoch),
                            torch.var(number_neighbor_iteration), torch.var(number_neighbor_epoch))

                for tag, value in loss_print.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        return inputs, pids

    def set_model_train(self):
        self.model.train()

        # Fix first BN
        fixed_bns = []
        for idx, (name, module) in enumerate(self.model.module.named_modules()):
            if name.find("layer3") != -1:
                break
            if name.find("bn") != -1:
                fixed_bns.append(name)
                module.eval()
