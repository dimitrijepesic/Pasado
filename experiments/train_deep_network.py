"""Trains deep8 (784 - 8 x (1024, sigmoid) - 10) with the recipe of Section_5_4/train_mnist.py
(results/deep_network_2026-11/PROTOCOL.md). Resumable: a checkpoint per epoch in --out-dir.

  python experiments/train_deep_network.py --out-dir DIR [--epochs 100] [--device cuda]
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deep8(depth=8, width=1024):
    mods = [nn.Flatten(), nn.Linear(784, width), nn.Sigmoid()]
    for _ in range(depth - 1):
        mods += [nn.Linear(width, width), nn.Sigmoid()]
    mods.append(nn.Linear(width, 10))
    return nn.Sequential(*mods)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-batches", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device(args.device)
    data = os.path.join(ROOT, "Section_5_4", "MNIST_Data")
    trainset = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=transforms.Compose(
        [transforms.RandomCrop(28, padding=4), transforms.ToTensor()]))
    loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    net = deep8().to(dev)
    opt = optim.Adam(net.parameters(), lr=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.75)
    ckpt = os.path.join(args.out_dir, "checkpoint.pt")
    start_epoch, log = 0, []
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=dev)
        net.load_state_dict(state["net"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng"]); start_epoch, log = state["epoch"], state["log"]
        print(f"resuming after epoch {start_epoch}")
    crit = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, args.epochs):
        t0, total, seen = time.time(), 0.0, 0
        for b, (x, y) in enumerate(loader):
            if args.limit_batches and b >= args.limit_batches:
                break
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0); seen += x.size(0)
        sched.step()
        log.append(dict(epoch=epoch + 1, train_loss=total / seen, seconds=time.time() - t0))
        print(f"epoch {epoch + 1}: loss {total / seen:.4f} ({time.time() - t0:.1f} s)", flush=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        rng=torch.get_rng_state(), epoch=epoch + 1, log=log), ckpt)
    torch.save(net.state_dict(), os.path.join(args.out_dir, "model_deep8.pth"))
    testset = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=transforms.ToTensor())
    net.eval()
    correct = []
    with torch.no_grad():
        for i in range(len(testset)):
            x, y = testset[i]
            if int(net(x.unsqueeze(0).to(dev)).argmax()) == y:
                correct.append(i)
    torch.save(torch.tensor(correct), os.path.join(args.out_dir, "indices_deep8.pth"))
    acc = len(correct) / len(testset)
    json.dump(dict(test_accuracy=acc, epochs=args.epochs, device=str(dev), log=log, torch=torch.__version__),
              open(os.path.join(args.out_dir, "training.json"), "w"), indent=1)
    print(f"test accuracy {acc:.4f}" + ("" if acc >= 0.9 else "  (below 90%: not used for bounds, see PROTOCOL.md)"))


if __name__ == "__main__":
    main()
