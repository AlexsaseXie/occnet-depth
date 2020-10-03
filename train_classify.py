import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
from classify.img_resnet import ImgClassify_ResNet18
from classify.depth_resnet import DepthClassify_Resnet18
from classify.pointnet import PointcloudClassify_Pointnet
from classify.dataset import get_dataset
from im2mesh import data

out_dir = 'out/classify/depthpc_world_512_origin_subdivision'
dataset_root = '/home3/data/xieyunwei/occnet_data/ShapeNet.with_depth/'
batch_size = 256
c_dim = 512

save_every = 1000
backup_every = 4000
lr_drop = 15000
quit_after = 20000
input_type = 'depth_pointcloud'
pretrained = True
absolute_depth = True
pred_with_img = True
depth_pointcloud_transfer = 'world'

train_dataset = get_dataset(dataset_root, 'train', input_type, absolute_depth=absolute_depth)
val_dataset = get_dataset(dataset_root, 'val', input_type, absolute_depth=absolute_depth)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, num_workers=4, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
if input_type == 'img':
    model = ImgClassify_ResNet18(13, c_dim=c_dim, pretrained=pretrained)
elif input_type == 'img_with_depth':
    model = DepthClassify_Resnet18(13, c_dim=c_dim, pretrained=pretrained, with_img=pred_with_img)
elif input_type == 'depth_pointcloud':
    model = PointcloudClassify_Pointnet(13, c_dim=c_dim, depth_pointcloud_transfer=depth_pointcloud_transfer)
else:
    raise NotImplementedError
model = model.to(device)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

try:
    pretrained_dict = torch.load(os.path.join(out_dir,'model.pt')).state_dict()
except :
    pretrained_dict = dict()

model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

epoch = 0
it = 0

optimizer = optim.Adam(model.parameters(), lr=1e-4)
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

def get_correctness():
    model.eval()
    correct_count = 0
    total_count = 0

    for val_batch in val_loader:
        data = val_batch

        inputs = data.get('inputs')
        current_batch_size = inputs.shape[0]

        out = model(data, device).to(device)
        class_gt = data.get('category').to(device)
        class_predict = out.max(dim=1)[1]
        
        #print('gt:',class_gt)
        #print('pred:',class_predict)
        correct_count += (class_predict == class_gt).sum().item()
        total_count += current_batch_size

    print('correct_count:',correct_count)
    print('total_count:',total_count)
    return correct_count / total_count

max_correctness = 0.
while True:
    epoch += 1
    for batch in train_loader: 
        model.train()
        optimizer.zero_grad()
        loss = model.get_loss(batch, device)
        loss.backward()
        optimizer.step()

        logger.add_scalar('train/loss', loss, it)
        print('[%d] loss: %f' %( it , loss.cpu() ) )
        
        if (it % save_every) == 0:
            current_correctness = get_correctness()

            print('[%d val] correctness: %f' % (it, current_correctness))
            if current_correctness > max_correctness:
                max_correctness = current_correctness
                output_best_path = os.path.join(out_dir,'model_best.pt')
                torch.save(model.state_dict(), output_best_path)
                output_best_path = os.path.join(out_dir,'encoder_best.pt')
                torch.save(model.features.state_dict(), output_best_path)
                if input_type.startswith('img'):
                    output_best_path = os.path.join(out_dir,'resnet18_best.pt')
                    torch.save(model.features.features.state_dict(), output_best_path)

            output_path = os.path.join(out_dir,'model.pt')
            torch.save(model.state_dict(), output_path)

        if (it % backup_every) == 0:
            output_path = os.path.join(out_dir,'model_%d.pt' % it)
            torch.save(model.state_dict(), output_path)

        if it == lr_drop:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5

        if it == quit_after:
            print('exiting!')
            exit(3)

        it += 1
            


