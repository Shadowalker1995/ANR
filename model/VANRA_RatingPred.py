import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np


class VANRA_RatingPred(nn.Module):
    """
    Aspect-Based Rating Predictor, using Aspect-based Representations & the estimated Aspect Importance
    """
    def __init__(self, logger, args, num_users, num_items):
        super(VANRA_RatingPred, self).__init__()

        self.logger = logger
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        # Dropout for the User & Item Aspect-Based Representations
        if self.args.dropout_rate > 0.0:
            self.userAspRepDropout = nn.Dropout(p=self.args.dropout_rate)
            self.itemAspRepDropout = nn.Dropout(p=self.args.dropout_rate)
            self.userVisRepDropout = nn.Dropout(p=self.args.dropout_rate)
            self.itemVisRepDropout = nn.Dropout(p=self.args.dropout_rate)

        # Global Offset/Bias (Trainable)
        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad=True)   # 1 x 1

        # User & Item Offset/Bias
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
    [Input]    userAspRep:  bsz x num_aspects x h1
    [Input]    itemAspRep:  bsz x num_aspects x h1
    [Input]    userAspImpt: bsz x num_aspects
    [Input]    itemAspImpt: bsz x num_aspects
    [Input]    userVisAttn: bsz x output_size
    [Input]    itemVisAttn: bsz x output_size
    [Input]    batch_uid:   bsz
    [Input]    batch_iid:   bsz
    [Output]   rating_pred: bsz x 1
    '''
    def forward(self, userAspRep, itemAspRep,
                userVisAttn, itemVisAttn, batch_uid, batch_iid, verbose=0):
        if verbose > 0:
            tqdm.write("\n\n**************************************** Aspect-Based Rating Predictor ****************************************")
            tqdm.write("[Input] userAspRep: {}".format(userAspRep.size()))          # bsz x num_aspects x h1
            tqdm.write("[Input] itemAspRep: {}".format(itemAspRep.size()))          # bsz x num_aspects x h1
            # tqdm.write("[Input] userAspImpt: {}".format(userAspImpt.size()))        # bsz x num_aspects
            # tqdm.write("[Input] itemAspImpt: {}".format(itemAspImpt.size()))        # bsz x num_aspects
            tqdm.write("[Input] userVisAttn: {}".format(userVisAttn.size()))        # bsz x output_size
            tqdm.write("[Input] itemVisAttn: {}".format(itemVisAttn.size()))        # bsz x output_size
            tqdm.write("[Input] batch_uid:  {}".format(batch_uid.size()))           # bsz
            tqdm.write("[Input] batch_iid:  {}".format(batch_iid.size()))           # bsz

        # User & Item Bias
        batch_userOffset = self.uid_userOffset(batch_uid)
        batch_itemOffset = self.iid_itemOffset(batch_iid)

        if verbose > 0:
            tqdm.write("\nbatch_userDocOffset: {}".format(batch_userOffset.size()))  # bsz x 1
            tqdm.write("batch_itemDocOffset: {}".format(batch_itemOffset.size()))    # bsz x 1

        # Dropout for the User & Item Aspect-Based Representations
        if self.args.dropout_rate > 0.0:
            userAspRep = self.userAspRepDropout(userAspRep)
            itemAspRep = self.itemAspRepDropout(itemAspRep)
            userVisAttn = self.userVisRepDropout(userVisAttn)
            itemVisAttn = self.itemVisRepDropout(itemVisAttn)

        lstAspRating = []

        # (bsz x num_aspects x h1) -> (num_aspects x bsz x h1)
        userAspRep = torch.transpose(userAspRep, 0, 1)                              # num_aspects x bsz x h1
        itemAspRep = torch.transpose(itemAspRep, 0, 1)                              # num_aspects x bsz x h1

        # normalize
        userAspRep = F.normalize(userAspRep, p=2, dim=2)
        itemAspRep = F.normalize(itemAspRep, p=2, dim=2)

        for k in range(self.args.num_aspects):
            aspRating = torch.sum(torch.mul(userAspRep[k], itemAspRep[k]), 1, keepdim=True)
            if verbose > 0 and k == 0:
                tqdm.write("\taspRating: {}".format(aspRating.size()))              # bsz x 1

            lstAspRating.append(aspRating)

        # List of (bsz x 1) -> (bsz x num_aspects)
        DocRating = torch.cat(lstAspRating, dim=1)
        if verbose > 0:
            tqdm.write("\nrating_pred: {} ('Raw' Aspect-Level Ratings)".format(DocRating.size()))     # bsz x num_aspects

        # # Multiply Each Aspect-Level (Predicted) Rating with the Corresponding User-Aspect Importance & Item-Aspect Importance
        # # (bsz x num_aspects) * (bsz x num_aspects) * (bsz x num_aspects) -> bsz x num_aspects
        # rating_pred = torch.mul(torch.mul(userAspImpt, itemAspImpt), rating_pred)                       # bsz x num_aspects
        # if verbose > 0:
        #     tqdm.write("rating_pred: {} (Multiplied with User-Aspect Importance & Item-Aspect Importance)".format(rating_pred.size()))

        # Sum over all Aspects
        DocRating = torch.sum(DocRating, dim=1, keepdim=True)                   # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Summed over All {} Aspects)".format(DocRating.size(), self.args.num_aspects))

        # normalize
        userVisAttn = F.normalize(userVisAttn, p=2, dim=1)
        itemVisAttn = F.normalize(itemVisAttn, p=2, dim=1)

        # userVisAttn: bsz x output_size
        VisRating = torch.sum(torch.mul(userVisAttn, itemVisAttn), 1, keepdim=True)
        if verbose > 0:
            tqdm.write("\n\tVisRating: {}".format(VisRating.size()))                # bsz x 1

        rating_pred = (DocRating + VisRating)/2
        # rating_pred = DocRating
        # rating_pred = VisRating
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include Visual Rating Predict)".format(rating_pred.size()))    # bsz x 1

        # Include User & Item Doc Bias
        rating_pred = rating_pred + batch_userOffset + batch_itemOffset             # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include User & Item Bias)".format(rating_pred.size()))

        # Include Global Bias
        rating_pred = rating_pred + self.globalOffset                               # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include Global Bias)".format(rating_pred.size()))

        if verbose > 0:
            tqdm.write("\n[VANRA_RatingPred Output] rating_pred: {}".format(rating_pred.size()))
            tqdm.write("**************************************** ***************************** ****************************************\n")

        return rating_pred
