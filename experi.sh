
set -e
set -u
set -x

python Convolution.py --epochs 10 --lr 0.001 --gb 1 --valid_set_size 0.7 --name withgb20

python Convolution.py --epochs 10 --lr 0.001 --rot 1 --name withrot20 --valid_set_size 0.7

#python Convolution.py --epochs 20 --lr 0.001 --spk 1 --name withspk20 --valid_set_size 0.7

#python Convolution.py --epochs 20 --lr 0.001 --isw 1 --name withisw20 --valid_set_size 0.7

python Convolution.py --epochs 10 --lr 0.001 --shr 1 --name withshr20 --valid_set_size 0.7

python Convolution.py --epochs 10 --lr 0.001 --noaug 0 --gb 0 --name withoutaug --valid_set_size 0.7


python Convolution.py --epochs 10 --lr 0.001 --gb 1 --shr 1  --name withgbshr20 --valid_set_size 0.7
