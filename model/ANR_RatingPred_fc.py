import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np


class ANR_RatingPred(nn.Module):
    """
    Aspect-Based Rating Predictor, using Aspect-based Representations & the estimated Aspect Importance
    """
    def __init__(self, logger, args, num_users, num_items):
        super(ANR_RatingPred, self).__init__()

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

        # Dimensionality of the abstract user & item representations
        # Well, this can also be a hyperparameter, but we simply set it to h1
        self.user_item_rep_dim = self.args.h1

        self.fcLayer = nn.Sequential(
            # bsz x (num_aspects x h1) -> # bsz x h1
            nn.Linear(self.args.num_aspects * self.args.h1, self.user_item_rep_dim),
            nn.ReLU(),
            nn.Dropout(self.args.dropout_rate),
        )

        self.prediction = nn.Linear(2 * self.user_item_rep_dim, 1)

    '''
    [Input]    userAspRep:  bsz x num_aspects x h1
    [Input]    itemAspRep:  bsz x num_aspects x h1
    [Input]    userAspImpt: bsz x num_aspects
    [Input]    itemAspImpt: bsz x num_aspects
    [Input]    batch_uid:   bsz
    [Input]    batch_iid:   bsz
    [Output]   rating_pred: bsz x 1
    '''
    def forward(self, userAspRep, itemAspRep, userAspImpt, itemAspImpt, batch_uid, batch_iid, verbose=0):
        if verbose > 0:
            tqdm.write("\n\n**************************************** Aspect-Based Rating Predictor ****************************************")
            tqdm.write("[Input] userAspRep: {}".format(userAspRep.size()))          # bsz x num_aspects x h1
            tqdm.write("[Input] itemAspRep: {}".format(itemAspRep.size()))          # bsz x num_aspects x h1
            tqdm.write("[Input] userAspImpt: {}".format(userAspImpt.size()))        # bsz x num_aspects
            tqdm.write("[Input] itemAspImpt: {}".format(itemAspImpt.size()))        # bsz x num_aspects
            tqdm.write("[Input] batch_uid:  {}".format(batch_uid.size()))           # bsz
            tqdm.write("[Input] batch_iid:  {}".format(batch_iid.size()))           # bsz

        userAspImpt = userAspImpt.unsqueeze(2)                                      # bsz x num_aspects x 1
        itemAspImpt = itemAspImpt.unsqueeze(2)                                      # bsz x num_aspects x 1

        userAspRep = torch.mul(userAspRep, userAspImpt)                             # bsz x num_aspects x h1
        itemAspRep = torch.mul(itemAspRep, itemAspImpt)                             # bsz x num_aspects x h1

        # Concatenate all aspect-level representations into a single vector
        userAspRep = userAspRep.view(-1, self.args.num_aspects * self.args.h1)      # bsz x (num_aspects x h1)
        itemAspRep = itemAspRep.view(-1, self.args.num_aspects * self.args.h1)      # bsz x (num_aspects x h1)
        # if verbose > 0:
        #     tqdm.write("\n[Concatenated] concatUserRep: {}".format(concatUserRep.size()))
        #     tqdm.write("[Concatenated] concatItemRep: {}".format(concatItemRep.size()))

        userAspRep = self.fcLayer(userAspRep)                                      # bsz x h1
        itemAspRep = self.fcLayer(itemAspRep)                                      # bsz x h1

        # Concatenate the user & item representations for prediction
        userItemRep = torch.cat((userAspRep, itemAspRep), 1)                        # bsz x (h1 x 2)
        if verbose > 0:
            tqdm.write("\n[Input to Final Prediction Layer] userItemRep: {}".format(userItemRep.size()))
        rating_pred = self.prediction(userItemRep)                                  # bsz x 1
        if verbose > 0:
            tqdm.write("\nrating_pred: {} ('Raw' Ratings)".format(rating_pred.size()))

        # User & Item Bias
        batch_userOffset = self.uid_userOffset(batch_uid)
        batch_itemOffset = self.iid_itemOffset(batch_iid)

        # Include User Bias & Item Bias
        rating_pred = rating_pred + batch_userOffset + batch_itemOffset             # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include User & Item Bias)".format(rating_pred.size()))

        # Include Global Bias
        rating_pred = rating_pred + self.globalOffset                               # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include Global Bias)".format(rating_pred.size()))

        if verbose > 0:
            tqdm.write("\n[ANR_RatingPred Output] rating_pred: {}".format(rating_pred.size()))
            tqdm.write(
                "============================== =================================== ==============================\n")

        return rating_pred
