#!/bin/sh

time python3 get_lipschitz.py --network 3layer
time python3 get_lipschitz.py --network 4layer
time python3 get_lipschitz.py --network 5layer
time python3 get_lipschitz.py --network big
