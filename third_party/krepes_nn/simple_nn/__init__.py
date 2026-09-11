from .train_NN import train
from .ae_dnn import train_AE_NN, AE_loss_nn
from .loss_functions import BT_loss, VICReg_loss, SimCLRLoss, spectral_loss, \
    simple_contrastive_loss, krr_nystrom_loss, byol_loss_fn
from .model import CNN_sup, FC_Supervised, TinyTransformer, Resnet, BYOL, Projector
