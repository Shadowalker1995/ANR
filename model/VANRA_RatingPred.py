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

        # Global Offset/Bias (Trainable)
        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad=True)   # b(0) 1 x 1

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
    def forward(self, userAspRep, itemAspRep, userAspImpt, itemAspImpt,
                userVisAttn, itemVisAttn, batch_uid, batch_iid, verbose=0):
        if verbose > 0:
            tqdm.write("\n\n**************************************** Aspect-Based Rating Predictor ****************************************")
            tqdm.write("[Input] userAspRep: {}".format(userAspRep.size()))          # bsz x num_aspects x h1
            tqdm.write("[Input] itemAspRep: {}".format(itemAspRep.size()))          # bsz x num_aspects x h1
            tqdm.write("[Input] userAspImpt: {}".format(userAspImpt.size()))        # bsz x num_aspects
            tqdm.write("[Input] itemAspImpt: {}".format(itemAspImpt.size()))        # bsz x num_aspects
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

        # =========== Dropout for the User & Item Aspect-Based Representations ===========
        if self.args.dropout_rate > 0.0:
            userAspRep = self.userAspRepDropout(userAspRep)
            itemAspRep = self.itemAspRepDropout(itemAspRep)

            if verbose > 0:
                tqdm.write("\n[After Dropout (Dropout Rate of {:.1f})] userAspRep: {}".format(self.args.dropout_rate, userAspRep.size()))
                tqdm.write("[After Dropout (Dropout Rate of {:.1f})] itemAspRep: {}".format(self.args.dropout_rate, itemAspRep.size()))
        # =========== Dropout for the User & Item Aspect-Based Representations ===========

        lstAspRating = []

        # (bsz x num_aspects x h1) -> (num_aspects x bsz x h1)
        userAspRep = torch.transpose(userAspRep, 0, 1)                              # num_aspects x bsz x h1
        itemAspRep = torch.transpose(itemAspRep, 0, 1)                              # num_aspects x bsz x h1

        for k in range(self.args.num_aspects):
            user_Doc = torch.unsqueeze(userAspRep[k], 1)
            item_Doc = torch.unsqueeze(itemAspRep[k], 2)
            if verbose > 0 and k == 0:
                tqdm.write("\nAs an example, the aspect-level rating prediction for <Aspect {}>:\n".format(k))
                tqdm.write("\tuser_Doc: {}".format(user_Doc.size()))                # bsz x 1 x h1
                tqdm.write("\titem_Doc: {}".format(item_Doc.size()))                # bsz x h1 x 1

            # (bsz x 1 x h1) x (bsz x h1 x 1) -> bsz x 1 x 1
            aspRating = torch.matmul(user_Doc, item_Doc)
            if verbose > 0 and k == 0:
                tqdm.write("\n\taspRating: {}".format(aspRating.size()))            # bsz x 1 x 1

            aspRating = torch.squeeze(aspRating, 2)
            if verbose > 0 and k == 0:
                tqdm.write("\taspRating: {}".format(aspRating.size()))              # bsz x 1

            lstAspRating.append(aspRating)

        # List of (bsz x 1) -> (bsz x num_aspects)
        rating_pred = torch.cat(lstAspRating, dim=1)
        if verbose > 0:
            tqdm.write("\nrating_pred: {} ('Raw' Aspect-Level Ratings)".format(rating_pred.size()))     # bsz x num_aspects

        # Multiply Each Aspect-Level (Predicted) Rating with the Corresponding User-Aspect Importance & Item-Aspect Importance
        # (bsz x num_aspects) * (bsz x num_aspects) * (bsz x num_aspects) -> bsz x num_aspects
        rating_pred = userAspImpt * itemAspImpt * rating_pred                       # bsz x num_aspects
        if verbose > 0:
            tqdm.write("rating_pred: {} (Multiplied with User-Aspect Importance & Item-Aspect Importance)".format(rating_pred.size()))

        # Sum over all Aspects
        rating_pred = torch.sum(rating_pred, dim=1, keepdim=True)                   # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Summed over All {} Aspects)".format(rating_pred.size(), self.args.num_aspects))

        # userVisAttn: bsz x output_size
        user_Vis = torch.unsqueeze(userVisAttn, 1)
        item_Vis = torch.unsqueeze(itemVisAttn, 2)
        if verbose > 0:
            tqdm.write("\nThe rating prediction of Visual Part:\n")
            tqdm.write("\tuser_Vis: {}".format(user_Vis.size()))                    # bsz x 1 x output_size
            tqdm.write("\titem_Vis: {}".format(item_Vis.size()))                    # bsz x output_size x 1
        VisRating = torch.matmul(user_Vis, item_Vis)
        if verbose > 0 and k == 0:
            tqdm.write("\n\tVisRating: {}".format(VisRating.size()))                # bsz x 1 x 1
        VisRating = torch.squeeze(VisRating, 2)
        if verbose > 0 and k == 0:
            tqdm.write("\tVisRating: {}".format(VisRating.size()))                  # bsz x 1
        rating_pred = rating_pred + VisRating
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
            tqdm.write("\n[ANR_RatingPred Output] rating_pred: {}".format(rating_pred.size()))
            tqdm.write("**************************************** ***************************** ****************************************\n")

        return rating_pred
