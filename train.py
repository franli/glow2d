"""
MLE Training of Glow[1], on 2D dataset
[1] https://arxiv.org/abs/1807.03039
"""
import argparse
import os
import torch
from model import Flow
import util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


device = torch.device('cuda')

parser = argparse.ArgumentParser(description='MLE Training of Glow (2D)')
parser.add_argument('--epoch', default=200, type=int, help='number of training epochs')
parser.add_argument('--batch', default=100, type=int, help='batch size')
parser.add_argument('--dataset', default='8gaussians', type=str, help='2D dataset to use') # '8gaussians', '2spirals', 'checkerboard', 'rings', 'pinwheel'
parser.add_argument('--samples', default=10000, type=int, help='number of 2D samples for training')

parser.add_argument('--width', default=128, type=int, help='width of the glow model') 
parser.add_argument('--depth', default=10, type=int, help='depth of the glow model') 
parser.add_argument('--n_levels', default=1, type=int, help='levels of the glow model') 

parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--resume', type=bool, default=False, help='Resume from checkpoint')
args = parser.parse_args()


# ------------------------------
# I. MODEL
# ------------------------------
flow = Flow(width=args.width, depth=args.depth, n_levels=args.n_levels, data_dim=2).to(device)

# ------------------------------
# II. OPTIMIZER
# ------------------------------
optim_flow = torch.optim.Adam(flow.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# ------------------------------
# III. DATA LOADER
# ------------------------------
dataset, dataloader = util.get_data(args)

# ------------------------------
# IV. TRAINING
# ------------------------------
def main(args):
    start_epoch = 0
    if args.resume:
        print('Resuming from checkpoint at ckpts/flow.pth.tar...')
        checkpoint = torch.load('ckpts/flow.pth.tar')
        flow.load_state_dict(checkpoint['flow'])
        start_epoch = checkpoint['epoch'] + 1
    for epoch in range(start_epoch, start_epoch + args.epoch):
        for i, x in enumerate(dataloader):           
            x = x.to(device)

            optim_flow.zero_grad() 
            loss_flow = - flow.log_prob(x).mean()
            loss_flow.backward()
            optim_flow.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [Loss: %f]"
                % (epoch, start_epoch + args.epoch, i, len(dataloader), loss_flow.item())

            )

        print('Saving flow model to ckpts/flow.pth.tar...')
        state = {
        'flow': flow.state_dict(),
        'value': loss_flow,
        'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/flow.pth.tar')

        # visualization
        util.plot(dataset, flow, epoch, device)







if __name__ == '__main__':
    print(args)
    main(args)