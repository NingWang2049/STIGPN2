import argparse
import os
import shutil
import time
import numpy as np
import json
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from callbacks import AverageMeter, Logger
from data_utils.data_loader_frames import VideoFolder
from utils import save_results
import datetime
from utils import send_wx_msg
import random
random.seed(0)
parser = argparse.ArgumentParser(description='PyTorch Smth-Else')

# Path related arguments
parser.add_argument('--root_frames', type=str, default='/raid/dataset/20BN-something/video/frames', help='path to the folder with frames')
parser.add_argument('--json_data_train', type=str, default='dataset_splits/compositional/train.json', help='path to the json file with train video meta data')
parser.add_argument('--json_data_val', type=str, default='dataset_splits/compositional/validation.json', help='path to the json file with validation video meta data')
parser.add_argument('--json_file_labels', type=str, default='dataset_splits/compositional/labels.json', help='path to the json file with ground truth labels')
parser.add_argument('--img_feature_dim', default=256, type=int, metavar='N',
                    help='intermediate feature dimension for image-based features')
parser.add_argument('--coord_feature_dim', default=256, type=int, metavar='N',
                    help='intermediate feature dimension for coord-based features')
parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                    metavar='W', help='gradient norm clipping (default: 5)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--k', default=1, type=int, metavar='N',
                    help='kernel size')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=72, type=int,
                    metavar='N', help='mini-batch size (default: 72)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[24, 35, 45], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print_freq', '-p', default=40, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', '-l', default=10, type=int,
                    metavar='N', help='frequency to write in tensorboard (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes', default=174, type=int,
                    help='num of class in the model')
parser.add_argument('--num_boxes', default=4, type=int,
                    help='num of boxes for each image')
parser.add_argument('--num_frames', default=16, type=int,
                    help='num of frames for the model')
parser.add_argument('--dataset', default='smth_smth',
                    help='which dataset to train')
parser.add_argument('--logdir', default='./logs',
                    help='folder to output tensorboard logs')
parser.add_argument('--logname', default='exp',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--ckpt', default='model/pretrained_weights/kinetics-res50.pth',
                    help='folder to output checkpoints')
parser.add_argument('--fine_tune', help='path with ckpt to restore')
parser.add_argument('--tracked_boxes', type=str, default='/home/wn/datasets/Something-Something/annotations/bounding_box_annotations.json', help='choose tracked boxes')
parser.add_argument('--shot', default=5)
parser.add_argument('--restore_i3d')
parser.add_argument('--restore_custom')

best_loss = 1000000

import pynvml

men_large = 0
current_gpu_id = 0
pynvml.nvmlInit()
for gpu_id in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if men_large < meminfo.free:
        men_large = meminfo.free
        current_gpu_id = gpu_id
pynvml.nvmlShutdown()

def main():
    global args, best_loss
    args = parser.parse_args()
    print(args)
    # create model
    os.environ['CUDA_VISIBLE_DEVICES'] = str(current_gpu_id)
    from models.Net import SSTGCN,V_SSTGCN
    model1 = SSTGCN(args).cuda()
    model1.load_state_dict(torch.load('./checkpoints/SSTGCN_max_score_model.pkl'))
    model2 = V_SSTGCN(args).cuda()
    model2.load_state_dict(torch.load('./checkpoints/V_SSTGCN_max_score_model.pkl'))
    if args.start_epoch is None:
        args.start_epoch = 0
    
    #model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    val_boxes_path = '/ssd/datasets/Something-Else/val.json'
    print('... Loading box annotations might take a minute ...')
    with open(val_boxes_path, 'r') as f:
        val_box_annotations = json.load(f)
    
    # create validation dataset
    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              frames_duration=args.num_frames,
                              args=args,
                              is_val=True,
                              if_augment=True,
                              model=None,
                              box_annotations = val_box_annotations,
                              )

    # create validation loader
    val_loader = torch.utils.data.DataLoader(
        dataset_val, #drop_last=True,
        batch_size=2*args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    validate(val_loader, model1, model2, class_to_idx=dataset_val.classes_dict)

def validate(val_loader, model1, model2, epoch=None, tb_logger=None, class_to_idx=None):
    global args
    batch_time = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    # switch to evaluate mode
    model1.eval()
    model2.eval()
    
    end = time.time()
    temp = open('results.txt', 'w')
    for i, (global_img_tensors,node_features, box_tensors, box_categories, video_label, video_label_text, video_id) in enumerate(val_loader):
        global_img_tensors,node_features, box_categories, box_tensors = global_img_tensors.cuda(),node_features.cuda(), box_categories.cuda(), box_tensors.cuda()
        # compute output

        with torch.no_grad():
            output1 = model1(global_img_tensors,node_features, box_categories, box_tensors).view((-1, len(val_loader.dataset.classes)))
            output2 = model2(global_img_tensors,node_features, box_categories, box_tensors).view((-1, len(val_loader.dataset.classes)))
            output = output1 + output2
            #if video_label == torch.argmax(output[0].cpu()):
            #    temp.writelines(video_id[0] + ' ' + video_label_text[0] + ' ' + 'R' + '\n')
            #else:
            #    temp.writelines(video_id[0] + ' ' + video_label_text[0] + ' ' + 'F' + '\n')
            acc1, acc5 = accuracy(output.cpu(), video_label, topk=(1, 5))

        # measure accuracy and record loss
        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})'.format(
                i, len(val_loader), batch_time=batch_time, acc_top1=acc_top1, acc_top5=acc_top5))
    temp.close()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
