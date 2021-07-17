import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import model
import dataset
import torchvision
from tensorboardX import SummaryWriter
import os

print("dataloading...")
trainset = dataset.MyDataset()
trainset = DataLoader(trainset, batch_size=512, shuffle=True)
print("dataload success")

network = model.Soccer().cuda().train()

epoch = 500
lr = 0.003
momentum = 0.9

if os.path.exists("runs"):
    os.system("rm -r ./runs")
writer = SummaryWriter('runs')


optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, network.parameters()), lr=lr, momentum=momentum)
step_size = int(epoch * 0.3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1 )
for epoch_i in range(1, epoch+1):
    total_loss = 0 
    for imgs, labels in trainset:
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        out = network(imgs)
        loss = F.mesloss(out, labels)
        loss.backword()
        optimizer.step()
        total_loss += loss
        print("epoch: ", epoch_i, ", train_loss:%.4f"%loss)
    
    scheduler.step()
    writer,add_scalar('loss', total_loss, global_step=epoch_i)

    if(total_loss<flag):
        flag = total_loss
        torch.save(network, "./weight/best.pth")
 
torch.save(network, "./weight/last.pth")
