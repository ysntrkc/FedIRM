CUDA_LAUNCH_BLOCKING=1 python test_main.py --dataset isic2019 --mode ssfl --rounds 101 --local_ep 5
CUDA_LAUNCH_BLOCKING=1 python test_main.py --dataset ham10000 --mode ssfl --rounds 101 --local_ep 5
CUDA_LAUNCH_BLOCKING=1 python test_main.py --dataset rsna --mode ssfl --rounds 101 --local_ep 5