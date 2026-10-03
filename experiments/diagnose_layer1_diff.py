"""Debugging script from an earlier investigation. It tracked down a 1e-6 difference
caused by float32 roots in the first vectorized boundary check (fixed by
promoting the roots to the input dtype). It should now report differences near 0."""
import os, sys, numpy as np, torch
torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(HERE, "..", "Section_5_4"))
from SimpleZono import *
from Duals import *
import precise_transformer as pt
from precise_transformer import AffineDualZonotope
from model import FCN
import torchvision, torchvision.transforms as transforms

REPO = os.path.join(HERE, "..")
net = FCN(3); net.load_state_dict(torch.load(os.path.join(REPO,'Section_5_4/trained/model_3layer.pth'), map_location='cpu')); net.eval()
testset = torchvision.datasets.MNIST(root=os.path.join(REPO,'Section_5_4/MNIST_Data'), train=False, download=False, transform=transforms.ToTensor())
ci = torch.load(os.path.join(REPO,'Section_5_4/trained/indices_3layer.pth'), map_location='cpu')
img=None
for i,(im,_) in enumerate(torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False)):
    if i in ci: img=im.flatten(); break

eps=2.0
ie=DualIntervalTensor(real_l=torch.tensor([0.0]),real_u=torch.tensor([eps])); ie.e1_l[0]=1; ie.e1_u[0]=1
ze=HyperDualIntervalToDualZonotope(ie); hazed=(torch.tensor([1])-ze)*img+ze

# Reproduce sigmoid_prime_product_tensor's setup for layer 1's x,y dual zonotopes.
np.random.seed(12345); torch.manual_seed(12345)
aff = AffineDualZonotope(hazed, net.fc1.weight.T)+net.fc1.bias.data
from SimpleZono import SigmoidZonotope
sig_real = SigmoidZonotope(aff.real)
rn = sig_real.get_num_noise_symbs()
if 0 < rn - aff.dual.get_num_noise_symbs():
    aff.dual.expand(rn - aff.dual.get_num_noise_symbs())
x = aff.real; y = aff.dual
lx,ux,ly,uy = x.get_lb(),x.get_ub(),y.get_lb(),y.get_ub()
xys = pt.get_linspace(lx,ux,ly,uy)
zs = [pt.sigmoid_prime_times_y(xy) for xy in xys]
lineqs = [pt.lin_reg_tensor(xys[i], zs[i]) for i in range(len(zs))]
ABCs = [(lineqs[i][1], lineqs[i][2], lineqs[i][0]) for i in range(len(lineqs))]
ABCs = [(a[0] + np.random.normal(0, .0001), a[1], a[2]) for a in ABCs]
As=torch.tensor([a[0] for a in ABCs]); Bs=torch.tensor([a[1] for a in ABCs]); Cs=torch.tensor([a[2] for a in ABCs])
ABCs_t=torch.stack((As,Bs,Cs),dim=1)

# Compare corner checks
c_old = torch.stack(pt.check_corners(ABCs, lx.clone(),ux.clone(),ly.clone(),uy.clone()))
c_new = pt.check_corners_tensor(ABCs_t.clone(), lx.clone(),ux.clone(),ly.clone(),uy.clone())
dc = (c_old-c_new).abs()
print(f"CORNERS   max|old-new| = {dc.max().item():.3e}   at neuron {dc.argmax().item()}")

# Compare boundary checks (note: these mutate inputs, so clone per call)
b_old_ly, b_old_uy = pt.check_nonlinear_boundary(lx.clone(),ux.clone(),ly.clone(),uy.clone(), ABCs)
b_new_ly, b_new_uy = pt.check_nonlinear_boundary_tensor(lx.clone(),ux.clone(),ly.clone(),uy.clone(), ABCs_t.clone())
b_old_ly=torch.stack(b_old_ly); b_old_uy=torch.stack(b_old_uy)
dly=(b_old_ly-b_new_ly).abs(); duy=(b_old_uy-b_new_uy).abs()
print(f"BOUND_ly  max|old-new| = {dly.max().item():.3e}   at neuron {dly.argmax().item()}")
print(f"BOUND_uy  max|old-new| = {duy.max().item():.3e}   at neuron {duy.argmax().item()}")

# Zoom into the worst boundary neuron
j = int(duy.argmax().item())
print(f"\n--- worst BOUND_uy neuron {j} ---")
print(f"  A={As[j].item():.6e} B={Bs[j].item():.6e} C={Cs[j].item():.6e}")
print(f"  lx={lx[j].item():.6e} ux={ux[j].item():.6e} ly={ly[j].item():.6e} uy={uy[j].item():.6e}")
print(f"  old_uy={b_old_uy[j].item():.10e}  new_uy={b_new_uy[j].item():.10e}")

# Recompute roots for that neuron both ways to see where they diverge
Aj=As[j:j+1];
A_by_uy = Aj/uy[j:j+1]
A_by_uy = pt.inf_to_nan(A_by_uy.clone())
l=-1.0/(6.0*np.sqrt(3.0)); u=-l
A_by_uy = pt.filter_range(l,u,A_by_uy.clone())
r1,r2,r3 = pt.inverse_sigmoid_2nd_deriv(A_by_uy.clone())
print(f"  A/uy={ (Aj/uy[j:j+1]).item():.6e}  roots(before lx,ux filter) = {r1.item():.6e},{r2.item():.6e},{r3.item():.6e}")
r1f=pt.filter_range(lx[j:j+1].clone(),ux[j:j+1].clone(),r1.clone())
r2f=pt.filter_range(lx[j:j+1].clone(),ux[j:j+1].clone(),r2.clone())
r3f=pt.filter_range(lx[j:j+1].clone(),ux[j:j+1].clone(),r3.clone())
print(f"  roots(after filter) = {r1f.item():.6e},{r2f.item():.6e},{r3f.item():.6e}")
