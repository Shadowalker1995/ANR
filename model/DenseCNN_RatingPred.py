import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np


class DenseCNN_RatingPred(nn.Module):
    def __init__(self, logger, args, num_users, num_items):
        super(DenseCNN_RatingPred, self).__init__()

        self.logger = logger
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        # Global Offset/Bias (Trainable)
        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad=True)   # 1 x 1

        # User Offset/Bias & Item Offset/Bias
        self.uid_userOffset = nn.Embedding(self.num_users, 1)                   # num_users x 1
        self.uid_userOffset.weight.requires_grad = True

        self.iid_itemOffset = nn.Embedding(self.num_items, 1)                   # num_items x 1
        self.iid_itemOffset.weight.requires_grad = True

        # Initialize Global Bias with 0
        self.globalOffset.data.fill_(0)

        # Initialize All User/Item Offset/Bias with 0
        self.uid_userOffset.weight.data.fill_(0)
        self.iid_itemOffset.weight.data.fill_(0)

    '''
    [Input]    batch_userFea:    bsz x output_size
    [Input]    batch_itemFea:    bsz x output_size
    [Input]    batch_uid:   bsz
    [Input]    batch_iid:   bsz
    [Output]   rating_pred: bsz x 1
    '''
    def forward(self, batch_userFea, batch_itemFea, batch_uid, batch_iid, verbose=0):
        if verbose > 0:
            tqdm.write("\n\n**************************************** DAttn Rating Predictor ****************************************")
            tqdm.write("[Input] batch_userFea: {}".format(batch_userFea.size()))              # bsz x output_size
            tqdm.write("[Input] batch_itemFea: {}".format(batch_itemFea.size()))              # bsz x output_size
            tqdm.write("[Input] batch_uid:  {}".format(batch_uid.size()))           # bsz
            tqdm.write("[Input] batch_iid:  {}".format(batch_iid.size()))           # bsz

        # User & Item Bias
        batch_userOffset = self.uid_userOffset(batch_uid)
        batch_itemOffset = self.iid_itemOffset(batch_iid)

        if verbose > 0:
            tqdm.write("\nbatch_userOffset: {}".format(batch_userOffset.size()))    # bsz x 1
            tqdm.write("batch_itemOffset: {}".format(batch_itemOffset.size()))      # bsz x 1

        rating_pred = torch.sum(torch.mul(batch_userFea, batch_itemFea), 1, keepdim=True)
        if verbose > 0:
            tqdm.write("\nrating_pred: {} ('Raw' Ratings)".format(rating_pred.size()))     # bsz x 1

        # Include User Bias & Item Bias
        rating_pred = rating_pred + batch_userOffset + batch_itemOffset             # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include User & Item Bias)".format(rating_pred.size()))

        # Include Global Bias
        rating_pred = rating_pred + self.globalOffset                               # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include Global Bias)".format(rating_pred.size()))

        if verbose > 0:
            tqdm.write("\n[DAttn_RatingPred Output] rating_pred: {}".format(rating_pred.size()))
            tqdm.write("**************************************** ***************************** ****************************************\n")

        return rating_pred
