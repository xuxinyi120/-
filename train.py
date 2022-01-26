#from LoadUCF101DataByTorch import trainset_loader, testset_loader
from dataset import VideoDataset
from model import LRCN
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

EPOCH = 500
LEARNING_RATE = 0.05
MOMENTUM = 0.9
GAMMA = 0.5
STEP_SIZE = 1
batch_size = 128
#checkpoint
resume_epoch = 0  # 继续时开始的epoch



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

dataset='mine'

model = LRCN().to(device)


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=STEP_SIZE,
#     gamma=GAMMA
# )



def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

modelName = 'LRCN'
# 加载模型参数，进行模型参数初始化和优化器参数初始化
if resume_epoch == 0:
    print("Training {} from scratch...".format(modelName))
else:
    checkpoint = torch.load(os.path.join('/mnt/nvme1n1/xxy/LRCN_PyTorch/model/checkpoint-epoch-7.pth.tar'),
                map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
        os.path.join('/mnt/nvme1n1/xxy/LRCN_PyTorch/model/checkpoint-epoch-7.pth.tar')))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_dict'])

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
model.to(device)
#criterion.to(device)

#加载数据集
print('Training model on {} dataset...'.format(dataset))
# 加载数据集
train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=128, shuffle=False, num_workers=4)
#val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=128, num_workers=4)
test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=128, num_workers=4)
print(len(train_dataloader.dataset))
print(len(test_dataloader.dataset))

def train(epoch):
    iteration = 0
    loss_plt=[]
    


    for i in range(epoch):
        model.train()
        # print('current lr', scheduler.get_last_lr())
        correct = 0

        for index, data in enumerate(train_dataloader):
            video_clips, label = data

            video_clips = video_clips.to(device)
            label = label.to(device)

            batch = video_clips.size(0)

            optimizer.zero_grad()

            output = model(video_clips)

            loss = F.cross_entropy(output, label)

            loss_plt.append(loss.item())

            loss.backward()
            optimizer.step()

            max_value, max_index = output.max(1, keepdim=True)
            correct += max_index.eq(label.view_as(max_index)).sum().item()

            iteration += 1
            #if (index+1)%20==0:
            print("Epoch:", i, "/", epoch-1, "\tIteration:", index, "/", len(train_dataloader)-1, "\tLoss: " + str(loss.item())+ "\n")
            #print("len(train_dataloader.dataset):" + str(len(train_dataloader.dataset)) + "\n")
            print("train_Accuracy: " + str((correct * 1.0 * 100) / batch_size)+ "\n")
            print("correct:" + str(correct)+ "\n")
            with open('/mnt/nvme1n1/xxy/LRCN/train_log.txt', 'a') as f:
                f.write("Epoch: " + str(i) + "/" + str(epoch-1) + "\tIteration:" + str(index) + "/" + str(len(train_dataloader)-1) + "\tLoss: " + str(loss.item()) + "\n")
                f.write("train_Accuracy: " + str((correct * 1.0 * 100) / batch_size)+ "\n")
                f.write("correct:"+str(correct)+ "\n")
            #correct = 0

       
        #save_checkpoint('model/checkpoint-%i.pth' % iteration, model, optimizer)
        #print("Epoch:", i, "/", epoch-1,  "\tLoss: " + str(loss.item())+"\n")
        #print("train_Accuracy: " + str((correct * 1.0 * 100) / len(train_dataloader.dataset))+"\n")
        #with open('/mnt/nvme1n1/xxy/LRCN/train_log.txt', 'a') as f:
        #    f.write("Epoch: " + str(i) + "/" + str(epoch-1) +  "\tLoss: " + str(loss.item()) + "\n")
        #    f.write("Accuracy: " + str((correct * 1.0 * 100) / len(train_dataloader.dataset))+ "\n")
        #    f.write("correct:"+str(correct)+ "\n"+"\n")
        
        test(i)    

    #save_checkpoint('model/checkpoint-%i.pth' % iteration, model, optimizer)

    if i % 100 == 0:
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
            }, os.path.join('/mnt/nvme1n1/xxy/LRCN/model', 'checkpoint-epoch-' + str(i) + '.pth.tar'))
        print("Save model at {}\n".format(os.path.join('model', 'checkpoint-' + str(i) + '.pth.tar')))

    plt.figure()
    plt.plot(loss_plt)
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('')
    plt.show()



def test(i_epoch):
    model.eval()

    correct = 0

    with torch.no_grad():
        for index, data in enumerate(test_dataloader):
            video_clips, label = data

            video_clips = video_clips.to(device)
            label = label.to(device)

            output = model(video_clips)

            max_value, max_index = output.max(1, keepdim=True)
            correct += max_index.eq(label.view_as(max_index)).sum().item()

    print("test_correct:"+ str(correct)  + "\n")
    #print("len(test_dataloader.dataset)" + str(len(test_dataloader.dataset)) + "\n")
    print("test_Accuracy: " + str(correct * 1.0 * 100 / len(test_dataloader.dataset))+ "\n")
    with open('/mnt/nvme1n1/xxy/LRCN/test_log.txt', 'a') as f:
        f.write("Epoch " + str(i_epoch) + "'s Accuracy: " + str(correct * 1.0 * 100 / len(test_dataloader.dataset)) + "\n")


if __name__ == '__main__':
    train(EPOCH)
