#!/usr/bin/python
# -*- coding: utf-8 -*-
# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import sys
import xml.etree.ElementTree as ET

# Our Modules
reader_folder = os.path.realpath(os.path.abspath('..'))
if reader_folder not in sys.path:
    sys.path.append(reader_folder)
import utils
import numpy as np
from MPNN import MPNN
from LogMetric import AverageMeter, Logger
from graph_reader import read_cxl
from Plotter import Plotter
import scipy.io, scipy.stats
# from tensorboardX import SummaryWriter
import torch.utils.data as data
from tqdm import tqdm


torch.multiprocessing.set_sharing_strategy('file_system')

def collate_g(batch):

    batch_sizes = np.max(np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
                                len(list(input_b[2].values())[0])]
                                if input_b[2] else
                                [len(input_b[1]), len(input_b[1][0]), 0,0]
                                for (input_b, target_b, _) in batch]), axis=0)

    if batch_sizes[0] == 1:
        batch_sizes[0] = 2
    if batch_sizes[3] == 0:
        batch_sizes[3] = 100
    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))

    target = np.zeros((len(batch), len(batch[0][1])))
    index = np.zeros((len(batch)))

    for i in range(len(batch)):

        num_nodes = len(batch[i][0][1])
        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][0][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][0][1]

        # Edges
        for edge in batch[i][0][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][0][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][0][2][edge]

        # Target
        target[i, :] = batch[i][1]
        index[i] = batch[i][2]

    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    target = torch.FloatTensor(target)

    return (g, h, e), target, index

def vectorized_bspline_coeff(vi, vs):
    C = np.zeros(vi.shape)
    sel1 = np.logical_and((vs >= vi), (vs < vi + 1))
    C[sel1] = (1 / 6) * np.power(vs[sel1] - vi[sel1], 3)

    sel2 = np.logical_and((vs >= vi + 1), (vs < vi + 2))
    C[sel2] = (1 / 6) * (
                -3 * np.power(vs[sel2] - vi[sel2] - 1, 3) + 3 * np.power(vs[sel2] - vi[sel2] - 1, 2) + 3 * (
                    vs[sel2] - vi[sel2] - 1) + 1)

    sel3 = np.logical_and((vs >= vi + 2), (vs < vi + 3))
    C[sel3] = (1 / 6) * (3 * np.power(vs[sel3] - vi[sel3] - 2, 3) - 6 * np.power(vs[sel3] - vi[sel3] - 2, 2) + 4)

    sel4 = np.logical_and((vs >= vi + 3), (vs < vi + 4))
    C[sel4] = (1 / 6) * np.power(1 - (vs[sel4] - vi[sel4] - 3), 3)

    return C


nland = 5


def get_bsplines_matrix(neval=10):
    lb = 2
    ub = nland + 1
    sval = np.linspace(lb, ub, neval)
    ns = len(sval)
    sval = sval[:, np.newaxis]
    S = np.repeat(sval, nland, axis=1)
    L = nland
    I = np.repeat(np.arange(L)[np.newaxis, :], ns, axis=0)
    A = vectorized_bspline_coeff(I, S)
    sumA = A.sum(1)
    Coef = A / np.repeat(sumA[:,np.newaxis], L, 1)
    return Coef


def reconstruct_stroke(splines, mean, scale, coef, n):
    s = np.zeros([n, 2])
    s[:,0] = np.dot(coef, splines[:, 0])
    s[:, 1] = np.dot(coef, splines[:, 1])
    s = s / scale
    s = s + mean
    return s

def kp2stroke(strokeinfo):
    n = len(strokeinfo)
    connection = np.zeros([n, n])
    for i in range(n):
        node_ind = strokeinfo[i]
        con = np.logical_or(strokeinfo == node_ind[0], strokeinfo == node_ind[1])
        con = np.logical_or(con[:,0], con[:,1])
        connection[i, np.where(con)[0]] = 1
        if node_ind[0] == node_ind[1]:
            connection[i, i] = 1
        else:
            connection[i, i] = 0
    return connection

def reconstruct_img(points):
    imgs = []
    xx = np.arange(2, 28, 7)
    yy = np.arange(2, 28, 7)
    xv, yv = np.meshgrid(xx, yy)
    index = np.concatenate((yv.reshape([16, 1]), xv.reshape([16, 1])), axis=1)
    xx1 = np.arange(4)
    yy1 = np.arange(4)
    xv1, yv1 = np.meshgrid(xx1, yy1)
    index1 = np.concatenate((yv1.reshape([16, 1]), xv1.reshape([16, 1])), axis=1)
    for p in points:
        img = np.zeros([4, 4])
        dis = scipy.stats.multivariate_normal(p, 3)
        img[index1[:,0], index1[:,1]] = dis.pdf(index)
        imgs.append(img)
    return np.array(imgs).reshape([-1,16])


class MNIST(data.Dataset):
    def __init__(self, root, number_clusters=20, is_train = True, type='kp'):
        if is_train:
            info = 'skeleton_train.mat'
        else:
            info = 'skeleton_test.mat'
        info = os.path.join(root, info)
        # scaler = StandardScaler()
        info = scipy.io.loadmat(info)
        G = info['G']
        G = np.concatenate((G, np.arange(G.shape[1]).reshape([1, -1])), axis=0)
        s = []
        r = []
        label = []
        image = []
        stroke = []
        index = []
        relation = []
        adj = []
        loc = []
        mean = []
        pm = []
        stroke_point_num = []
        global_index = []
        N = len(G[0, :])
        k = 0
        for i in range(N):
            spline = G[2, i]

            if len(spline) == 0:
                continue
            global_index.append(G[8, i])
            normalize_info = G[3, i]
            image.append(G[0, i])
            label.append(G[1, i][0][0])
            adj.append(G[6, i])
            relation.append(G[7, i])
            # loc.append((G[5,i] - [14, 14]) / 14.)
            # pm.append(reconstruct_img(G[5,i]))
            for j in range(len(spline)):
                s.append(np.array(spline[j][0]).astype(np.float))
                r.append(np.array(normalize_info[2][0][j][0][0]).astype(np.float))
                mean.append(np.array(normalize_info[1][0][j][0][0]).astype(np.float))
                index.append(k)
                stroke.append(np.array(G[4, i][j][0]).astype(np.float))
                stroke_point_num.append(len(np.array(G[4, i][j][0])))
            k += 1
        index = np.array(index)
        label = np.array(label)
        s = np.array(s)
        s1 = s.reshape([-1, 10])
        # s = scaler.fit_transform(s1)
        r = np.array(r)
        r1 = r[:, 0]
        # r = r1 - r1.min() + 1e-5
        self.relation = relation
        self.r = r
        self.index = index
        self.label = label
        self.s = s1
        self.r = r1
        self.mean = np.array(mean)
        self.adj = adj
        self.G = G
        self.loc = loc
        self.train = is_train
        self.global_index = global_index
        # self.stroke_info = s1
        self.stroke_info = np.concatenate((s1, r1[:, np.newaxis]), axis=1)
        if type == 'stroke':
            self.stroke_info = np.concatenate((self.stroke_info, self.mean), axis=1)
        # print("valid data: " + str(len(label)))
        self.type = type
        self.pm = pm
        self.n = 5


        if self.type == 'stroke' or self.type == 'simple_stroke':
            coef = get_bsplines_matrix(self.n)
            stroke_relation = []
            node_info = []
            edge_info = []
            labels = []
            strokes = []
            weights = []
            for item in tqdm(range(len(self.label))):

                relation_ = kp2stroke(self.relation[item])
                g = np.asmatrix(relation_)
                ind = np.where(self.index == item)[0]
                labels.append(self.label[item])
                stroke_relation.append(g)
                h = self.stroke_info[ind]
                control_points = h[:, :10]
                scale_point = h[:, 10]
                mean_point = h[:, 11:]
                inner_dis = []
                inner_angle = []
                connecting_point = []
                for jj in range(len(ind)):
                    stroke = reconstruct_stroke(control_points[jj].reshape([5, 2]), mean_point[jj], scale_point[jj], coef, self.n)
                    strokes.append(stroke)
                    stroke1_1 = stroke[:, :1]
                    stroke1_2 = stroke[:, 1:]

                    # dot_product = np.dot(stroke1_1, stroke1_1.T) + np.dot(stroke1_2, stroke1_2.T)
                    # sqroot = np.sqrt(np.square(stroke1_1)+np.square(stroke1_2))
                    # sqroot_matrix = np.dot(sqroot, sqroot.T)
                    # angle = dot_product / sqroot_matrix
                    # angle = angle.flatten()
                    # inner_angle.append(angle)

                    distance_map = np.square(stroke1_1).repeat(self.n, 1) + np.square(stroke1_1.T).repeat(self.n, 0) - 2 * np.dot(
                        stroke1_1, stroke1_1.T)
                    distance_map += np.square(stroke1_2).repeat(self.n, 1) + np.square(stroke1_2.T).repeat(self.n,
                                                                                                      0) - 2 * np.dot(
                        stroke1_2, stroke1_2.T)
                    distance_map += 1e-13
                    distance_map = np.sqrt(distance_map)
                    distance_map = distance_map.flatten()
                    inner_dis.append(distance_map)

                inner_dis = np.array(inner_dis)
                # inner_angle = np.array(inner_angle)
                # h = np.concatenate((inner_dis, inner_angle), axis=1)

                h = inner_dis

                node_info.append(h)
                e = {}
                ee = np.where(relation_ == 1)
                # if len(ee[0]) == 0:
                #     e[(-1, -1)] = np.zeros([self.n*self.n]).flatten()
                for jj in range(len(ee[0])):
                    if ee[0][jj] >= ee[1][jj]:
                        stroke1 = reconstruct_stroke(control_points[ee[0][jj]].reshape([5, 2]), mean_point[ee[0][jj]],
                                                     scale_point[ee[0][jj]], coef, self.n)
                        stroke2 = reconstruct_stroke(control_points[ee[1][jj]].reshape([5, 2]), mean_point[ee[1][jj]],
                                                     scale_point[ee[1][jj]], coef, self.n)
                        stroke1_1 = stroke1[:, :1]
                        stroke1_2 = stroke1[:, 1:]
                        stroke2_1 = stroke2[:, :1]
                        stroke2_2 = stroke2[:, 1:]

                        # e[(ee[0][jj], ee[1][jj])] = angle
                        distance_map = np.square(stroke1_1).repeat(self.n, 1) + np.square(stroke2_1.T).repeat(self.n,
                                                                                                         0) - 2 * np.dot(
                            stroke1_1, stroke2_1.T)
                        distance_map += np.square(stroke1_2).repeat(self.n, 1) + np.square(stroke2_2.T).repeat(self.n,
                                                                                                          0) - 2 * np.dot(
                            stroke1_2, stroke2_2.T)
                        min_dist_point_ind = np.unravel_index(np.argmin(distance_map, axis=None), distance_map.shape)
                        min_dist_point = (stroke1[min_dist_point_ind[0]] + stroke2[min_dist_point_ind[1]])/2
                        connecting_point.append(min_dist_point)
                        distance_map += 1e-13
                        distance_map = np.sqrt(distance_map)
                        distance_map = distance_map.flatten()

                        # dot_product = np.dot(stroke1_1, stroke2_1.T) + np.dot(stroke1_2, stroke2_2.T)
                        # sqroot1 = np.sqrt(np.square(stroke1_1) + np.square(stroke1_2))
                        # sqroot2 = np.sqrt(np.square(stroke2_1) + np.square(stroke2_2))
                        # sqroot_matrix = np.dot(sqroot1, sqroot2.T)
                        # angle = dot_product / sqroot_matrix
                        # angle = angle.flatten()


                        e[(ee[0][jj], ee[1][jj])] = distance_map
                        # e[(ee[0][jj], ee[1][jj])] = np.concatenate([distance_map, angle])
                edge_info.append(e)
            self.labels = labels
            self.stroke_relation = stroke_relation
            self.node_info = node_info
            self.edge_info = edge_info
            self.strokes = strokes
            self.weights = np.array(weights)



    def __getitem__(self, item):
        if self.type == 'kp':
            g = self.adj[item]
            g = np.asmatrix(g)
            h = self.loc[item]
            ind = np.where(self.index == item)[0]
            # connection = self.r_ic[ind, :]
            connection = self.stroke_info[ind, :]
            e = {}
            ee = self.relation[item]
            for j, i in enumerate(ee):
                e_t = connection[j].tolist()
                e[(i[0]-1, i[1]-1)] = e_t
            target = [self.labels[item]]
            return (g, h, e), target, item
        elif self.type == 'stroke':
            g = self.stroke_relation[item]
            target = [self.labels[item]]
            h = self.node_info[item]
            e = self.edge_info[item]
            return (g, h, e), target, item
        elif self.type == 'simple_stroke':
            g = self.stroke_relation[item]
            target = [self.labels[item]]
            s = self.strokes[item]
            return (g, s), target, item
        else:
            g = []
            h = []
            e = []
            target = []
            return (g, h, e), target, item

    def __len__(self):
        return len(self.labels)

# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

class LinearModel(nn.Module):
    def __init__(self, model, out, target):
        super(LinearModel, self).__init__()
        self.model = model
        self.linear = nn.Linear(in_features=out, out_features=target)
        torch.nn.init.eye_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, g, h, e):
        x = self.model(g, h, e)
        # x = nn.ReLU()(x)
        x = self.linear(x)
        return nn.LogSoftmax()(x)

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--logPath', default='model/distance_linear_5_better', help='log path')
parser.add_argument('--plotLr', default=False, help='allow plotting the data')
parser.add_argument('--plotPath', default='model/distance_linear_5_better', help='plot path')
parser.add_argument('--resume', default=None,
                    help='path to latest checkpoint')
parser.add_argument('--savepath', default='model/distance_linear_5_better',
                    help='path to save checkpoint')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='Number of epochs to train (default: 360)')
parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 10]), default=0.0001, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.11, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
parser.add_argument('--schedule', type=list, default=[0.2, 0.3], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

best_acc1 = 0


def main():

    global args, best_acc1
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    global writer


    # Load data
    # root = args.datasetPath
    # subset = args.subSet

    print('Prepare files')

    # train_classes, train_ids = read_cxl(os.path.join(root, subset, 'train.cxl'))
    # test_classes, test_ids = read_cxl(os.path.join(root, subset, 'test.cxl'))
    # valid_classes, valid_ids = read_cxl(os.path.join(root, subset, 'validation.cxl'))
    #
    # class_list = list(set(train_classes + test_classes))
    # num_classes = len(class_list)
    data_train = MNIST(root='./', is_train=True, type='stroke')
    data_test = MNIST(root='./', is_train=False, type='stroke')
    
    # Define model and optimizer
    print('Define model')
    # Select one graph
    g_tuple, l, ind = data_train[0]
    g, h_t, e = g_tuple

    print('\tStatistics')
    stat_dict = utils.get_graph_stats(data_train, ['degrees'])

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                                               shuffle=True,
                                               collate_fn=collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    # valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size,
    #                                            collate_fn=utils.collate_g,
    #                                            num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size,
                                              collate_fn=collate_g,
                                              num_workers=args.prefetch, pin_memory=True)

    print('\tCreate model')
    model_base = MPNN([len(h_t[0]), len(list(e.values())[0])], hidden_state_size=25,
                 message_size=20, n_layers=1, l_target=10,
                         type='regression')


    print('Optimizer')


    criterion = nn.NLLLoss()

    evaluation = utils.accuracy

    print('Logger')
    logger = Logger(args.logPath)


    if args.plotLr:
        print('Plotter')
        plotter = Plotter(args.plotPath)

    lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])

    model = LinearModel(model_base, 10, 10)

    # get the best checkpoint if available without training
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            # args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {}; accuracy {})".format(best_model_file, checkpoint['epoch'],
                                                                             best_acc1))
        else:
            print("=> no best model found at '{}'".format(best_model_file))


    best_acc1 = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Check cuda')
    if args.cuda:
        print('\t* Cuda')
        model = model.cuda()
        criterion = criterion.cuda()



    # Epoch for loop
    for epoch in range(0, args.epochs):

        if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
            args.lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, evaluation, logger)

        # evaluate on test set
        acc1 = validate(test_loader, model, criterion, evaluation, logger, epoch)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc1': best_acc1,
                                'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.savepath)

        # Logger step
        logger.log_value('learning_rate', args.lr).step()
    # utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc1': best_acc1,
    #                           'optimizer': optimizer.state_dict(), }, is_best=False, directory='/home/scheng53/Robust_Active_Learning/skeleton/GNN/model1')

    # get the best checkpoint and test it with test set
    if args.savepath:
        checkpoint_dir = args.savepath
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {}; accuracy {})".format(best_model_file, checkpoint['epoch'],
                                                                             best_acc1))
            model.eval()
            _ = validate(test_loader, model, criterion, evaluation, logger)
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    # For testing
    # validate(test_loader, model, criterion, evaluation)
    # writer.close()


def train(train_loader, model, criterion, optimizer, epoch, evaluation, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, ((g, h, e), target, _) in enumerate(train_loader):
        
        # Prepare input data
        target = torch.squeeze(target).type(torch.LongTensor)
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Measure data loading time
        data_time.update(time.time() - end)

        def closure():
            optimizer.zero_grad()

            # Compute output
            output = model(g, h, e)
            train_loss = criterion(output, target)

            acc = Variable(evaluation(output.data, target.data, topk=(1,))[0])

            # Logs
            losses.update(train_loss.item(), g.size(0))
            accuracies.update(acc.item(), g.size(0))
            # compute gradient and do SGD step
            train_loss.backward()
            return train_loss

        optimizer.step(closure)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, acc=accuracies))
                          
    logger.log_value('train_epoch_loss', losses.avg)
    logger.log_value('train_epoch_accuracy', accuracies.avg)
    # writer.add_scalar('train_loss', losses.avg, epoch)
    # writer.add_scalar('train_acc', accuracies.avg, epoch)

    print('Epoch: [{0}] Average Accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, acc=accuracies, loss=losses, b_time=batch_time))


def validate(val_loader, model, criterion, evaluation, logger=None, epoch=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    index_all = []
    for i, ((g, h, e), target, index) in enumerate(val_loader):

        # Prepare input data
        target = torch.squeeze(target).type(torch.LongTensor)
        index_all.extend(index)
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Compute output
        output = model(g, h, e)

        # Logs
        test_loss = criterion(output, target)
        acc = Variable(evaluation(output.data, target.data, topk=(1,))[0])

        losses.update(test_loss.item(), g.size(0))
        accuracies.update(acc.item(), g.size(0))

    print(' * Average Accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(acc=accuracies, loss=losses))
    # np.save(os.path.join(args.resume, 'wrong_index'), np.array(index_all))
    # writer.add_scalar('test_loss', losses.avg, epoch)
    # writer.add_scalar('test_acc', accuracies.avg, epoch)
    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_accuracy', accuracies.avg)

    return accuracies.avg


def plot_examples(data_loader, model, epoch, plotter, ind = [0, 10, 20]):

    # switch to evaluate mode
    model.eval()

    for i, (g, h, e, target) in enumerate(data_loader):
        if i in ind:
            subfolder_path = 'batch_' + str(i) + '_t_' + str(int(target[0][0])) + '/epoch_' + str(epoch) + '/'
            if not os.path.isdir(args.plotPath + subfolder_path):
                os.makedirs(args.plotPath + subfolder_path)

            num_nodes = torch.sum(torch.sum(torch.abs(h[0, :, :]), 1) > 0)
            am = g[0, 0:num_nodes, 0:num_nodes].numpy()
            pos = h[0, 0:num_nodes, :].numpy()

            plotter.plot_graph(am, position=pos, fig_name=subfolder_path+str(i) + '_input.png')

            # Prepare input data
            if args.cuda:
                g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
            g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

            # Compute output
            model(g, h, e, lambda cls, id: plotter.plot_graph(am, position=pos, cls=cls,
                                                          fig_name=subfolder_path+ id))


    
if __name__ == '__main__':
    main()
