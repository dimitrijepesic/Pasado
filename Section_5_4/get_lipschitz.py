import sys

sys.path.insert(0, '../forward_mode_tensorized_src')

from SimpleZono import *
from Duals import *
from precise_transformer import *

import os
from timeit import default_timer as timer
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

torch.set_default_dtype(torch.float64)

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Get haze Lipschitz constant of MNIST Network')
parser.add_argument('--network', choices=['3layer', '4layer', '5layer', 'big'], help='neural network architecture',
                    default='3layer')
parser.add_argument("--no-save", action="store_true")
# Split [0, eps] into k pieces and take the max over them (default 1).
parser.add_argument("--n-splits", type=int, default=1)
# Run only some of the 16 epsilons, given by index.
parser.add_argument("--eps-indices", type=int, nargs="+", default=None,
                    help="indices into the default 16-epsilon range; default all")
# The precise transformer adds random noise to coefficient A, so two runs differ
# slightly. With a seed, np.random is reset before every precise forward pass and
# runs repeat exactly. Without one, the behaviour is unchanged.
parser.add_argument("--seed", type=int, default=None,
                    help="reseed np.random before every precise forward pass; default: no seeding")
parser.add_argument("--device", default=os.environ.get("PASADO_DEVICE", "cpu"),
                    help="torch device for the analysis, for example cpu or cuda; default: cpu")
parser.add_argument("--num-images", type=int, default=30,
                    help="number of correctly classified test images; default: 30")
parser.add_argument("--precise-only", action="store_true",
                    help="skip the plain zonotope and interval analyses")
args = parser.parse_args()

num_images_to_test = args.num_images
if num_images_to_test != 30:
    print(f'WARNING: --num-images {num_images_to_test}. The saved Pasado results are averages '
          f'over 30 images, so these values must not be compared with them.')

device = torch.device(args.device)
if device.type == 'cuda' and not torch.cuda.is_available():
    sys.exit('--device cuda was requested but CUDA is not available')


def now():
    """Timer reading. On CUDA, wait for queued kernels first so the time is real."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return timer()


test_transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST(root='./MNIST_Data', train=False,
                                     download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

from model import FCN, FCNBig

network = args.network
print(f'===== {network} Network =====')
print('config', dict(device=str(device), seed=args.seed, num_images=num_images_to_test,
                     n_splits=args.n_splits, eps_indices=args.eps_indices,
                     precise_only=args.precise_only,
                     selectors=dict(vectorized=USE_VECTORIZED_PRECISE,
                                    vec_boundary=USE_VECTORIZED_BOUNDARY,
                                    batched_lstsq=USE_BATCHED_LSTSQ,
                                    batched_grid=USE_BATCHED_GRID,
                                    real_cubic=USE_REAL_CUBIC),
                     torch=torch.__version__, threads=torch.get_num_threads()))

if network == '3layer':
    net = FCN(3)
    layers = 3
elif network == '4layer':
    net = FCN(4)
    layers = 4
elif network == '5layer':
    net = FCN(5)
    layers = 5
else:
    net = FCNBig()
net.load_state_dict(torch.load(f'trained/model_{network}.pth', map_location='cpu'))
net.eval()
net.to(device)

if 'big' == network:  # define forward functions for the "big" network
    def forward_zono(x):
        x = SigmoidDualZonotope(AffineDualZonotope(x, net[1].weight.T) + net[1].bias)
        x = SigmoidDualZonotope(AffineDualZonotope(x, net[3].weight.T) + net[3].bias)
        x = SigmoidDualZonotope(AffineDualZonotope(x, net[5].weight.T) + net[5].bias)
        x = SigmoidDualZonotope(AffineDualZonotope(x, net[7].weight.T) + net[7].bias)
        x = AffineDualZonotope(x, net[9].weight.T) + net[9].bias
        return x


    def forward_interval(x):
        x = Sigmoid_di(x @ abstract_di(net[1].weight.T) + abstract_di(net[1].bias))
        x = Sigmoid_di(x @ abstract_di(net[3].weight.T) + abstract_di(net[3].bias))
        x = Sigmoid_di(x @ abstract_di(net[5].weight.T) + abstract_di(net[5].bias))
        x = Sigmoid_di(x @ abstract_di(net[7].weight.T) + abstract_di(net[7].bias))
        x = x @ abstract_di(net[9].weight.T) + abstract_di(net[9].bias)
        return x


    def forward_zono_precise(x):
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net[1].weight.T) + net[1].bias)
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net[3].weight.T) + net[3].bias)
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net[5].weight.T) + net[5].bias)
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net[7].weight.T) + net[7].bias)
        x = AffineDualZonotope(x, net[9].weight.T) + net[9].bias
        return x

else:  # define forward functions for the "3/4/5-layer" networks
    def forward_zono(x):
        x = SigmoidDualZonotope(AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
        x = SigmoidDualZonotope(AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
        if layers >= 4:
            x = SigmoidDualZonotope(AffineDualZonotope(x, net.fc3.weight.T) + net.fc3.bias)
        if layers == 5:
            x = SigmoidDualZonotope(AffineDualZonotope(x, net.fc4.weight.T) + net.fc4.bias.data)
        x = AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data
        return x


    def forward_zono_precise(x):
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
        if layers >= 4:
            x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc3.weight.T) + net.fc3.bias.data)
        if layers == 5:
            x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc4.weight.T) + net.fc4.bias.data)
        x = AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data
        return x


    def forward_interval(x):
        x = Sigmoid_di(x @ abstract_di(net.fc1.weight.T) + abstract_di(net.fc1.bias.data))
        x = Sigmoid_di(x @ abstract_di(net.fc2.weight.T) + abstract_di(net.fc2.bias.data))
        if layers >= 4:
            x = Sigmoid_di(x @ abstract_di(net.fc3.weight.T) + abstract_di(net.fc3.bias.data))
        if layers == 5:
            x = Sigmoid_di(x @ abstract_di(net.fc4.weight.T) + abstract_di(net.fc4.bias.data))
        x = x @ abstract_di(net.fc_final.weight.T) + abstract_di(net.fc_final.bias.data)
        return x

correct_indices = torch.load(f'trained/indices_{network}.pth', map_location=torch.device('cpu'))

lc_zonos = []
lc_zonos_precise = []
lc_intervals = []
time_zonos = []
time_intervals = []
time_precise = []

n_splits = args.n_splits

test_range = [10 ** (-k / 4) * 2 for k in range(2, 18)]
if args.eps_indices is not None:
    test_range = [test_range[i] for i in args.eps_indices]
pbar_total = tqdm(total=len(test_range), position=0)

for epsilon in test_range:  # change back to 2,18
    print(f'Running epsilon={epsilon}')
    lc_zono_total = 0
    lc_zono_precise_total = 0
    lc_interval_total = 0

    time_zono_total = 0
    time_interval_total = 0
    time_precise_total = 0

    img_index = 0
    correct_images = 0
    used_ids = []

    delta = epsilon / n_splits
    interval_eps_l = []

    for i in range(n_splits):
        l_ = i * delta
        r_ = l_ + delta
        interval_eps = DualIntervalTensor(real_l=torch.tensor([l_], device=device),
                                          real_u=torch.tensor([r_], device=device))
        interval_eps.e1_l[0] = 1
        interval_eps.e1_u[0] = 1
        interval_eps_l.append(interval_eps)

    with torch.no_grad(), tqdm(total=num_images_to_test, position=1) as pbar_eps:
        for (image, _) in testloader:
            if img_index in correct_indices:
                img_f = image.flatten().to(device)
                one = torch.tensor([1], device=device)

                if not args.precise_only:
                    # zonotope
                    start = now()
                    lc_zono_temp = []
                    for i in range(n_splits):
                        zono_eps = HyperDualIntervalToDualZonotope(interval_eps_l[i])
                        hazed_zono = (one - zono_eps) * img_f + zono_eps
                        output = forward_zono(hazed_zono)
                        lc_zono_ = torch.max(
                            torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))).item()
                        lc_zono_temp.append(lc_zono_)

                    lc_zono = max(lc_zono_temp)
                    end = now()
                    time_zono_total += end - start
                    lc_zono_total += lc_zono

                # PRECISE Zonotope
                start = now()
                lc_zono_precise_temp = []
                for i in range(n_splits):
                    zono_eps = HyperDualIntervalToDualZonotope(interval_eps_l[i])
                    hazed_zono = (one - zono_eps) * img_f + zono_eps
                    if args.seed is not None:
                        np.random.seed(args.seed)
                    output = forward_zono_precise(hazed_zono)
                    lc_zono_precise_ = torch.max(
                        torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))).item()
                    lc_zono_precise_temp.append(lc_zono_precise_)
                lc_zono_precise = max(lc_zono_precise_temp)
                end = now()
                time_precise_total += end - start
                lc_zono_precise_total += lc_zono_precise

                if not args.precise_only:
                    # interval
                    start = now()
                    lc_interval_temp = []
                    for i in range(n_splits):
                        hazed_int = (1 - interval_eps_l[i]) * img_f + interval_eps_l[i]
                        output = forward_interval(hazed_int)
                        lc_interval_ = torch.max(torch.maximum(torch.abs(output.e1_l), torch.abs(output.e1_u))).item()
                        lc_interval_temp.append(lc_interval_)
                    lc_interval = max(lc_interval_temp)
                    end = now()
                    time_interval_total += end - start
                    lc_interval_total += lc_interval

                correct_images += 1
                used_ids.append(img_index)
                pbar_eps.update(1)

            if correct_images == num_images_to_test:
                break

            img_index += 1

    lc_zonos_precise.append(lc_zono_precise_total / correct_images)
    time_precise.append(time_precise_total / correct_images)
    if not args.precise_only:
        lc_zonos.append(lc_zono_total / correct_images)
        lc_intervals.append(lc_interval_total / correct_images)
        time_zonos.append(time_zono_total / correct_images)
        time_intervals.append(time_interval_total / correct_images)

    pbar_total.update(1)

pbar_total.close()

print('lc_precise', lc_zonos_precise)
if not args.precise_only:
    print('lc_zonos', lc_zonos)
print('time_precise', time_precise)
print('image_ids', used_ids)

if not args.no_save:
    os.system('mkdir -p results')
    # A run that differs from the default setup gets its own file names, so it can
    # never overwrite the reference results.
    sfx = ''
    if n_splits != 1:
        sfx += f'_split{n_splits}'
    if num_images_to_test != 30:
        sfx += f'_n{num_images_to_test}'
    if args.eps_indices is not None:
        sfx += '_eps' + '-'.join(str(i) for i in args.eps_indices)
    if args.seed is not None:
        sfx += f'_seed{args.seed}'
    if device.type != 'cpu':
        sfx += f'_{device.type}'
    if not args.precise_only:
        torch.save(lc_zonos, f'results/lc_zonos_{network}{sfx}.pth')
        torch.save(lc_intervals, f'results/lc_intervals_{network}{sfx}.pth')
    torch.save(lc_zonos_precise, f'results/lc_precise_{network}{sfx}.pth')

    if not args.precise_only:
        torch.save(time_zonos, f'results/time_zonos_{network}{sfx}.pth')
        torch.save(time_intervals, f'results/time_intervals_{network}{sfx}.pth')
    torch.save(time_precise, f'results/time_precise_{network}{sfx}.pth')
