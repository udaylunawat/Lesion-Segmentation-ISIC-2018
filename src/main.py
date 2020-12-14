import os
import torch
import math
# import visdom
import torch.utils.data as Data
import argparse
import numpy as np
from tqdm import tqdm

from distutils.version import LooseVersion
from Datasets.ISIC2018 import ISIC2018_dataset
from utils.transform import ISIC2018_transform

from Models.networks.network import Comprehensive_Atten_Unet

from utils.dice_loss import SoftDiceLoss, get_soft_label, val_dice_isic
from utils.dice_loss import Intersection_over_Union_isic
from utils.dice_loss import jaccard_isic, jaccard_coef

from utils.evaluation import AverageMeter
from utils.binary import assd
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

Test_Model = {'Comp_Atten_Unet': Comprehensive_Atten_Unet}

Test_Dataset = {'ISIC2018': ISIC2018_dataset}

Test_Transform = {'ISIC2018': ISIC2018_transform}


def train(train_loader, model, criterion, optimizer, args, epoch):
    losses = AverageMeter()

    model.train()
    for step, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image = x.float().cuda()
        target = y.float().cuda()

        output = model(image)                                      # model output

        target_soft = get_soft_label(target, args.num_classes)     # get soft label
        loss = criterion(output, target_soft, args.num_classes)    # the dice losses
        losses.update(loss.data, image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % (math.ceil(float(len(train_loader.dataset))/args.batch_size)) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                epoch, step * len(image), len(train_loader.dataset),
                100. * step / len(train_loader), losses=losses))

    print('The average loss:{losses.avg:.4f}'.format(losses=losses))
    return losses.avg



def valid_isic(valid_loader, model, criterion, optimizer, args, epoch, minloss):
    val_losses = AverageMeter()
    val_isic_dice = AverageMeter()
    val_isic_jaccard = AverageMeter()
    model.eval()
    for step, (t, k) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        image = t.float().cuda()
        target = k.float().cuda()

        output = model(image)                                             # model output
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_soft = get_soft_label(output_dis, args.num_classes)
        target_soft = get_soft_label(target, args.num_classes)            # get soft label

        val_loss = criterion(output, target_soft, args.num_classes)       # the dice losses
        val_losses.update(val_loss.data, image.size(0))

        isic = val_dice_isic(output_soft, target_soft, args.num_classes)  # the dice score
        val_isic_dice.update(isic.data, image.size(0))

        # SKY
        jaccard = jaccard_isic(output_soft, target_soft, args.num_classes) # the jaccard score
        val_isic_jaccard.update(jaccard.data, image.size(0))

        if step % (math.ceil(float(len(valid_loader.dataset)) / args.batch_size)) == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                epoch, step * len(image), len(valid_loader.dataset), 100. * step / len(valid_loader),
                losses=val_losses))

    print('The ISIC Mean Average Dice score: {isic.avg: .4f}; '
          'The Average Loss score: {loss.avg: .4f}'.format(
           isic=val_isic_dice, loss=val_losses))
    
    # SKY
    print('The ISIC Mean Average Jaccard score: {isic.avg: .4f}; '
        'The Average Loss score: {loss.avg: .4f}'.format(
        isic=val_isic_jaccard, loss=val_losses))

    if val_losses.avg < min(minloss):
        minloss.append(val_losses.avg)
        print(minloss)
        modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'
        print('the best model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)

    return val_losses.avg, val_isic_dice.avg



# def test_isic(test_loader, model, args):
#     isic_dice = []
#     isic_iou = []
#     isic_assd = []

#     modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'
#     if os.path.isfile(modelname):
#         print("=> Loading checkpoint '{}'".format(modelname))
#         checkpoint = torch.load(modelname)
#         # start_epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['state_dict'])
#         # optimizer.load_state_dict(checkpoint['opt_dict'])
#         print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
#     else:
#         print("=> No checkpoint found at '{}'".format(modelname))

#     model.eval()
#     for step, (img, lab) in tqdm(enumerate(test_loader), total=len(test_loader)):
#         image = img.float().cuda()
#         target = lab.float().cuda()

#         output = model(image)                                   # model output
#         output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
#         output_soft = get_soft_label(output_dis, args.num_classes)
#         target_soft = get_soft_label(target, args.num_classes)  # get soft label

#         label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
#         output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

#         isic_b_dice = val_dice_isic(output_soft, target_soft, args.num_classes)                # the dice accuracy
#         isic_b_iou = Intersection_over_Union_isic(output_soft, target_soft, args.num_classes)  # the iou accuracy
#         isic_b_asd = assd(output_arr[:, :, :, 1], label_arr[:, :, :, 1])                       # the assd

#         dice_np = isic_b_dice.data.cpu().numpy()
#         iou_np = isic_b_iou.data.cpu().numpy()
#         isic_dice.append(dice_np)
#         isic_iou.append(iou_np)
#         isic_assd.append(isic_b_asd)

#     isic_dice_mean = np.average(isic_dice)
#     isic_dice_std = np.std(isic_dice)

#     isic_iou_mean = np.average(isic_iou)
#     isic_iou_std = np.std(isic_iou)

#     isic_assd_mean = np.average(isic_assd)
#     isic_assd_std = np.std(isic_assd)
#     print('The ISIC mean Accuracy: {isic_dice_mean: .4f}; The Placenta Accuracy std: {isic_dice_std: .4f}'.format(
#            isic_dice_mean=isic_dice_mean, isic_dice_std=isic_dice_std))
#     print('The ISIC mean IoU: {isic_iou_mean: .4f}; The ISIC IoU std: {isic_iou_std: .4f}'.format(
#            isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std))
#     print('The ISIC mean assd: {isic_asd_mean: .4f}; The ISIC assd std: {isic_asd_std: .4f}'.format(
#            isic_asd_mean=isic_assd_mean, isic_asd_std=isic_assd_std))


def test_isic(test_loader, model, args):
    isic_dice = []
    isic_iou = []
    isic_assd = []
    isic_j3 = []
    # isic_j4 = []
    # isic_jf = []

    modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    model.eval()
    for step, (img, lab) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = img.float().cuda()
        target = lab.float().cuda()

        output = model(image)                                   # model output
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_soft = get_soft_label(output_dis, args.num_classes)
        target_soft = get_soft_label(target, args.num_classes)  # get soft label

        label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
        output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

        isic_b_dice = val_dice_isic(target_soft, output_soft, args.num_classes)                # the dice accuracy
        isic_b_iou = Intersection_over_Union_isic(output_soft, target_soft, args.num_classes)  # the iou accuracy
        isic_b_asd = assd(output_arr[:, :, :, 1], label_arr[:, :, :, 1])                       # the assd

        dice_np = isic_b_dice.data.cpu().numpy()
        iou_np = isic_b_iou.data.cpu().numpy()
        isic_dice.append(dice_np)
        isic_iou.append(iou_np)
        isic_assd.append(isic_b_asd)

        # SKY
        # isic_b_jaccard1_mean, isic_b_jaccard1_thresh = compute_jaccard(target_soft, output_soft)
        # isic_b_jaccard2iou = iou_numpy(labels = target_soft, outputs = output_soft)
        isic_b_jaccard3 = jaccard_coef(target_soft, output_soft) # jaccard_isic(output_soft, target_soft, args.num_classes)

        # isic_b_jaccard4 = jaccard_coef_loss(target_soft, output_soft)
        # isic_b_jaccardf = jaccard_isic(output_soft, target_soft, args.num_classes)

        # isic_jaccard_1_np = isic_b_jaccard1_mean.data.cpu().numpy()
        # isic_jaccard_2_np = isic_b_jaccard2iou.data.cpu().numpy()
        isic_jaccard_3_np = isic_b_jaccard3.data.cpu().numpy()
        # isic_jaccard_4_np = isic_b_jaccard4.data.cpu().numpy()
        # jaccard_np = isic_b_jaccardf.data.cpu().numpy()

        # if isic_jaccard_2_np >= 0.65:
        #     isic_j2.append(isic_jaccard_2_np)
        # else:
        #     isic_j2.append(0)
        # isic_j2.append(isic_jaccard_2_np)
        
        # isic_j1.append(isic_jaccard_1_np)
        # isic_j2.append(isic_jaccard_2_np)
        isic_j3.append(isic_jaccard_3_np)
        # isic_j4.append(isic_jaccard_4_np)
        # isic_jf.append(jaccard_np)

    isic_dice_mean = np.average(isic_dice)
    isic_dice_std = np.std(isic_dice)

    isic_iou_mean = np.average(isic_iou)
    isic_iou_std = np.std(isic_iou)

    isic_assd_mean = np.average(isic_assd)
    isic_assd_std = np.std(isic_assd)

    # SKY
    # isic_jaccard_mean1 = np.average(isic_j1)
    # isic_jaccard_mean2 = np.average(isic_j2)
    isic_jaccard_mean3 = np.average(isic_j3)
    isic_jaccard_std = np.std(isic_j3)
    # isic_jaccard_mean4 = np.average(isic_j4)
    # isic_jaccard_f_mean = np.average(isic_jf)



    print('The ISIC mean Accuracy: {isic_dice_mean: .4f};'.format(
           isic_dice_mean=isic_dice_mean))
    print('The ISIC mean IoU: {isic_iou_mean: .4f}; The ISIC IoU std: {isic_iou_std: .4f}'.format(
           isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std))
    print('The ISIC mean assd: {isic_asd_mean: .4f}; The ISIC assd std: {isic_asd_std: .4f}'.format(
           isic_asd_mean=isic_assd_mean, isic_asd_std=isic_assd_std))

    # SKY
    # print('The ISIC mean thresh Jaccard1: {isic_jaccard: .4f};'.format(
    #        isic_jaccard=isic_jaccard_mean1))
    # print('The ISIC mean thresh Jaccard2: {isic_jaccard: .4f};'.format(
    #        isic_jaccard=isic_jaccard_mean2))
    print('The ISIC mean thresh Jaccard: {isic_jaccard: .4f}; The ISIC Jaccard std: {isic_jaccard_std: .4f};'.format(
           isic_jaccard=isic_jaccard_mean3, isic_jaccard_std = isic_jaccard_std))
    # print('The ISIC mean thresh Jaccard: {isic_jaccard: .4f};'.format(
    #        isic_jaccard=isic_jaccard_f_mean))

def main(args):
    minloss = [1.0]
    start_epoch = args.start_epoch

    # loading the dataset
    print('loading the {0},{1},{2} dataset ...'.format('train', 'validation', 'test'))
    trainset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='train',
                                       transform=Test_Transform[args.data])
    validset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='validation',
                                       transform=Test_Transform[args.data])
    testset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                      transform=Test_Transform[args.data])

    trainloader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validloader = Data.DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    testloader = Data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print('Loading is done\n')

    # Define model
    if args.data == 'ISIC2018':
        args.num_input = 3
        args.num_classes = 2
        args.out_size = (224, 300)
    model = Test_Model[args.id](args, args.num_input, args.num_classes)

    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network')
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # collect the number of parameters in the network
    print("------------------------------------------")
    print("Network Architecture of Model AttU_Net:")
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    print(model)
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))
    print("------------------------------------------")

    # Define optimizers and loss function
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr_rate,
                                 weight_decay=args.weight_decay)    # optimize all model parameters
    criterion = SoftDiceLoss()
    scheduler = StepLR(optimizer, step_size=256, gamma=0.5)

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    print("Start training ...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        scheduler.step()
        train_avg_loss = train(trainloader, model, criterion, optimizer, args, epoch)

        if args.data == 'ISIC2018':
            val_avg_loss, val_isic_dice = valid_isic(validloader, model, criterion, optimizer, args, epoch, minloss)
 
        # save models
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                filename = args.ckpt + '/' + str(epoch) + '_' + args.data + '_checkpoint.pth.tar'
                print('the model will be saved at {}'.format(filename))
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
                torch.save(state, filename)

    print('Training Done! Start testing')
    if args.data == 'ISIC2018':
        test_isic(testloader, model, args)
    print('Testing Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='Comp_Atten_Unet',
                        help='a name for identitying the model. Choose from the following options: Unet')

    # Path related arguments
    parser.add_argument('--root_path', default='data/processed/ISIC2018_Task1_npy_all',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='saved_models',
                        help='folder to output checkpoints')

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=200, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')

    # other arguments
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')
    parser.add_argument('--out_size', default=(224, 300), help='the output image size')
    parser.add_argument('--val_folder', default='folder0', type=str,
                        help='which cross validation folder')

    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id)
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_' + args.data + '_checkpoint.pth.tar'

    main(args)
