
set -e
set -u
set -x

python Convolution.py --epochs 20 --lr 0.001 --gb 1

python Convolution.py --epochs 20 --lr 0.001 --rot 1

python Convolution.py --epochs 20 --lr 0.001 --spk 1

python Convolution.py --epochs 20 --lr 0.001 --isw 1

python Convolution.py --epochs 20 --lr 0.001 --shr 1

python Convolution.py --epochs 20 --lr 0.001 --noaug 0 --gb 0


python Convolution.py --epochs 80 --lr 0.001 --gb 1 --isw 1 

python resnet18_pretrained.py --lr 0.001 --noaug 0 --gb 0 --epochs 30

python resnet18_pretrained.py --lr 0.001 --noaug 1 --gb 1 --epochs 30

python resnet18_pretrained.py --lr 0.001 --noaug 1 --gb 1 --isw 1 --epochs 40



# Code to be added to create plots:

# - create lists for validation loss, train loss, validation accuracy, train accuracy append those values and plot them
# - create other plots that you might feel are useful
# - Note the test set accuracy for them
