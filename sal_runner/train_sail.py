import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib; matplotlib.use('Agg')
import sys
sys.path.append('./')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO


# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--learning_rate',type=float,default=1e-4,
                    help='Learning Rate.')
parser.add_argument('--start_number', type=int, default=0)

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = 1

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset('train', cfg)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, num_workers=0, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')


assert cfg['method'] in ('SAL', 'SAIL_S3')


# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
backup_every = cfg['training']['backup_every']
visualize_every = cfg['training']['visualize_every']
if 'exit_after' in cfg['training']:
    exit_after = cfg['training']['exit_after']
else:
    exit_after = 1e9


print_model_info = False
instance_id = -1
for batch in train_loader:
    instance_id += 1

    if instance_id < args.start_number:
        continue

    # initialize the model
    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    # Print model
    if print_model_info == False:
        nparameters = sum(p.numel() for p in model.parameters())
        print(model)
        print('Total number of parameters: %d' % nparameters)
        print_model_info = True

    model_output_dir = os.path.join(out_dir, '%d' % instance_id)
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)

    logger = SummaryWriter(os.path.join(model_output_dir, 'logs'))

    checkpoint_io = CheckpointIO(model_output_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt', strict=False, load_optimizer=True)
    except FileExistsError:
        load_dict = dict()

    # begin training
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)
    z_vec=load_dict.get('z_vec', None)

    # TODO: load z_vec 
    if z_vec is not None:
        batch['z'] = z_vec
        batch['center'] = load_dict.get('center')
        batch['length'] = load_dict.get('length')
        batch['r_tensor'] = load_dict.get('r_tensor')
        batch['t_tensor'] = load_dict.get('t_tensor')
    
    # init_z
    trainer.init_z(batch)
    trainer.init_training_points_record(batch)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 200000], gamma=0.1)
    if trainer.z_optimizer is not None:
        z_scheduler = optim.lr_scheduler.MultiStepLR(trainer.z_optimizer, milestones=[100000, 200000], gamma=0.1)
    else:
        z_scheduler = None

    # model_vis_dir = os.path.join(model_output_dir, 'vis')
    # if not os.path.exists(model_vis_dir):
    #     os.mkdir(model_vis_dir)
    # trainer.vis_dir = model_vis_dir

    while it <= exit_after:
        it += 1

        # schedule
        if it <= 80000:
            trainer.point_range = [0, 20000]
            trainer.point_sample = 10000
        else:
            trainer.point_range = [20000, 120000]
            trainer.point_sample = 10000

        # # only nearby
        # trainer.point_range = [20000, 120000]
        # trainer.point_sample = 20000

        # # all
        # trainer.point_range = None
        # trainer.point_sample = 20000

        # if 'surface_point_weight' in cfg['training']:
        #     trainer.surface_point_weight = cfg['training']['surface_point_weight']

        loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)

        scheduler.step()
        if z_scheduler is not None:
            z_scheduler.step()

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Instance %02d] it=%03d, loss=%f'
                    % (instance_id, it, loss))

        # need memorize tensors
        out_dict = trainer.cube_set_K.export()

        # Visualize 
        # if visualize_every > 0 and (it % visualize_every) == 0:
        #     print('Visualizing')
        #     trainer.visualize(batch)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', it=it, loss_val_best=metric_val_best, **out_dict)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, it=it, loss_val_best=metric_val_best, **out_dict)

        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.eval_step(batch, initialize_z=False)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                    % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', it=it, loss_val_best=metric_val_best, **out_dict)

    trainer.clear_z()
