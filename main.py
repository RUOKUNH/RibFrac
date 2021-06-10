import torch.utils.data as data
import os
import argparse
import torch
import torch.optim as optim
import numpy as np
import SimpleITK  as stk
from tqdm import tqdm, trange
from model import UNet
from loss import SoftDiceLoss
from UNetDataset import UNetTrainDataset
import transforms as tsfm
import torch.nn as nn
from predict import predict
import argparse
import torch.distributed as dist


# x_test = torch.randn(1,1,32,32,32)
# print(x_test.shape)
# encoder = EncoderBlock(1)
# x_test, h = encoder(x_test)
# print(x_test.shape)
# decoder = DecoderBlock(1)
# x_test = decoder(x_test, h)
# print(x_test.shape)
# loss = SoftDiceLoss()
# x_test = torch.sigmoid(torch.randn(1,1,32,32,32))
# target_test = torch.randint(0,2, x_test.size())
# l = loss(x_test, target_test)

def main(args):
    
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    # torch.cuda.set_device('cuda:0')
                          
    dir_path = os.path.join(args.output_root, 'checkpoints')
    
    train_data_root = os.path.join(args.input_root, 'train_data')
    val_data_root = os.path.join(args.input_root, 'val_data')
    test_data_root = os.path.join(args.input_root, 'test_data')
    train_label_root = os.path.join(args.input_root, 'train_label')
    val_label_root = os.path.join(args.input_root, 'val_label')
    
    
    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    
    train_dataset = UNetTrainDataset(train_data_root, train_label_root, transforms = transforms)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    train_loader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = False)

    val_dataset = UNetTrainDataset(val_data_root, val_label_root, transforms = transforms) 
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    val_loader = data.DataLoader(val_dataset, batch_size = args.batch_size)  

    model = UNet().cuda()
    device = torch.device("cuda")
    model = nn.DataParallel(model, device_ids = [0])
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    

    best = [0, np.inf]
    trigger = 0
    criterion = SoftDiceLoss()
    if args.test == False:
        for epoch in trange(args.num_epoch):
            train(model, optimizer, criterion, train_loader, device)
            loss = val(model, val_loader, criterion, device)
            
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch+1}
            torch.save(state, os.path.join(args.output_root, 'model_epoch_{}.pth'.format(epoch+1)))
            
            trigger += 1
            if loss < best[1]:
                print('Saving best model')
                torch.save(state, os.path.join(args.output_root, 'best_model.pth'))
                best[0] = epoch + 1
                best[1] = loss
                trigger = 0
            print("Best performance at epoch {} : {}".format(best[0], best[1]))
            if args.early_stop is not None:
                if trigger >= args.early_stop:
                    print("Early Stop")
                    break
            torch.cuda.empty_cache()
    else:

        restore_model_path = os.path.join(args.output_root, 'best_model.pth')
        
        model.load_state_dict(torch.load(restore_model_path)['net'])
        model.eval()
        
        save_path = os.path.join(args.output_root, 'test_label')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        predict(model, test_data_root, save_path, restore_model_path)
        
        # test(model, test_dataset, test_loader, device, criterion, save_path)
        
    
def train(model, optimizer, criterion, train_loader, device = None):
    model.train()
    for _, (datas, labels) in tqdm(enumerate(train_loader)):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        datas, labels = datas.float(), labels.float()
        datas = datas.reshape(datas.shape[0] * datas.shape[1], 
                          datas.shape[2], datas.shape[3], datas.shape[4], datas.shape[5])
        labels = labels.reshape(labels.shape[0] * labels.shape[1], 
                          labels.shape[2], labels.shape[3], labels.shape[4], labels.shape[5])
        # datas, labels = datas.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        datas, labels = datas.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        output = model(datas)
        
        loss = criterion(output, labels)
        
        del output
        del datas
        del labels

        loss.backward()
   
        optimizer.step()

        optimizer.zero_grad()
        
def val(model, val_loader, criterion, device = None):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, (datas, labels) in tqdm(enumerate(val_loader)):
            datas, labels = datas.float(), labels.float()
            datas = datas.reshape(datas.shape[0] * datas.shape[1], 
                              datas.shape[2], datas.shape[3], datas.shape[4], datas.shape[5])
            labels = labels.reshape(labels.shape[0] * labels.shape[1], 
                              labels.shape[2], labels.shape[3], labels.shape[4], labels.shape[5])
            datas, labels = datas.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # datas, labels = datas.to(device), labels.to(device)
            output = model(datas)
            val_loss += criterion(output, labels)

            del output
            del datas
            del labels
    
    return val_loss

# def test(model, test_dataset, test_loader, device, criterion, save_path):
#     model.eval()
#     with torch.no_grad():
#         for _, (datas, data_files) in tqdm(enumerate(test_loader)):
#             datas = datas.float()
#             datas = datas.to(device)
#             output = model(datas)
#             pred = torch.sigmoid(output)
#             pred_label = pred > 0.5
#             ImageInfo = test_dataset.GetImageInfo()
#             for i in range(output.size(0)):
#                 data_file = data_files[i]
#                 (origin, direction, space) = (ImageInfo[data_file+'_origin'],
#                                               ImageInfo[data_file+'_dir'],
#                                               ImageInfo[data_file+'_space'])
#                 label_file = "RibFrac1-label.nii.gz"
#                 label_file = data_file.split('-')[0]+'-'+label_file.split('-')[1]
#                 label_file = os.path.join(save_path, label_file)
#                 label = pred_label[i,0,:,:,:]
#                 label = stk.GetImageFromArray(label)
#                 label.SetOrigin(origin)
#                 label.SetDirection(direction)
#                 label.SetSpacing(space)
#                 stk.WriteImage(label, label_file)

# input_root = r"E:\Hw\ML_Project\RibFrac\dataset"
# output_root = r"E:\Hw\ML_Project\RibFrac\result"
# main(10, input_root, output_root, 1, (32,32,32), 1e-3, None, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of Unet')
    parser.add_argument('--input_root',
                        default='./input',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epoch',
                        default=100,
                        help='num of epochs of training',
                        type=int)
    parser.add_argument('--batch_size',
                        default = 5,
                        help = 'batch_size',
                        type = int)
    parser.add_argument('--lr',
                        default = 1e-3,
                        help = 'learning rate',
                        type = float)
    parser.add_argument('--early_stop',
                        default = None,
                        help = 'early stop trigger, default None',
                        type = int)
    parser.add_argument('--test',
                        default = False,
                        help = 'whether train or test',
                        type = bool)
    parser.add_argument('--local_rank', 
                        default=-1, 
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--device', 
                        default=None, 
                        type=int)
    args = parser.parse_args()
    main(args)
