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

        # Global Offset/Bias (Trainable)
        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad=True)  # 1 x 1

        # User Offset/Bias & Item Offset/Bias
        self.uid_userOffset = nn.Embedding(self.num_users, 1)  # num_users x 1
        self.uid_userOffset.weight.requires_grad = True

        self.iid_itemOffset = nn.Embedding(self.num_items, 1)  # num_items x 1
        self.iid_itemOffset.weight.requires_grad = True

        # Initialize Global Bias with 0
        self.globalOffset.data.fill_(0)

        # Initialize All User/Item Offset/Bias with 0
        self.uid_userOffset.weight.data.fill_(0)
        self.iid_itemOffset.weight.data.fill_(0)

        # Dimensionality of the abstract user & item representations
        # Well, this can also be a hyperparameter, but we simply set it to h1
        self.user_item_rep_dim = self.args.h1

        self.fcLayer1 = nn.Sequential(
            # bsz x (num_aspects x h1) -> bsz x h1
            nn.Linear(self.args.num_aspects * self.args.h1 + self.args.output_size, self.user_item_rep_dim),
            nn.ReLU(),
            nn.Dropout(self.args.dropout_rate),
        )

        # self.fcLayer2 = nn.Sequential(
        #     # bsz x output_size -> bsz x h1
        #     nn.Linear(self.args.output_size, self.user_item_rep_dim),
        #     nn.ReLU(),
        #     nn.Dropout(self.args.dropout_rate),
        # )
        #
        # # bsz x (4 x h1) -> bsz x 1
        # self.prediction = nn.Linear(4 * self.user_item_rep_dim, 1)

    '''
    [Input]    userAspRep:  bsz x num_aspects x h1
    [Input]    itemAspRep:  bsz x num_aspects x h1
    [Input]    userVisAttn: bsz x output_size
    [Input]    itemVisAttn: bsz x output_size
    [Input]    batch_uid:   bsz
    [Input]    batch_iid:   bsz
    [Output]   rating_pred: bsz x 1
    '''

    def forward(self, userAspRep, itemAspRep,
                userVisAttn, itemVisAttn, batch_uid, batch_iid, verbose=0):
        if verbose > 0:
            tqdm.write(
                "\n\n**************************************** Aspect-Based Rating Predictor ****************************************")
            tqdm.write("[Input] userAspRep: {}".format(userAspRep.size()))  # bsz x num_aspects x h1
            tqdm.write("[Input] itemAspRep: {}".format(itemAspRep.size()))  # bsz x num_aspects x h1
            tqdm.write("[Input] userVisAttn: {}".format(userVisAttn.size()))  # bsz x output_size
            tqdm.write("[Input] itemVisAttn: {}".format(itemVisAttn.size()))  # bsz x output_size
            tqdm.write("[Input] batch_uid:  {}".format(batch_uid.size()))  # bsz
            tqdm.write("[Input] batch_iid:  {}".format(batch_iid.size()))  # bsz

        # Concatenate all aspect-level representations into a single vector
        userAspRep = userAspRep.view(-1, self.args.num_aspects * self.args.h1)  # bsz x (num_aspects x h1)
        itemAspRep = itemAspRep.view(-1, self.args.num_aspects * self.args.h1)  # bsz x (num_aspects x h1)

        # bsz x (num_aspects x h1 + output_size)
        user_out = torch.cat((userAspRep, userVisAttn), 1)
        item_out = torch.cat((itemAspRep, itemVisAttn), 1)

        # userAspRep = self.fcLayer1(userAspRep)  # bsz x h1
        # itemAspRep = self.fcLayer1(itemAspRep)  # bsz x h1
        #
        # userVisAttn = self.fcLayer2(userVisAttn)  # bsz x h1
        # itemVisAttn = self.fcLayer2(itemVisAttn)  # bsz x h1
        #
        # # Concatenate the user & item representations for prediction
        # userItemRep = torch.cat((userAspRep, itemAspRep, userVisAttn, itemVisAttn), 1)  # bsz x (h1 x 4)
        # if verbose > 0:
        #     tqdm.write("\n[Input to Final Prediction Layer] userItemRep: {}".format(userItemRep.size()))

        user_out = self.fcLayer1(user_out)  # bsz x h1
        item_out = self.fcLayer1(item_out)  # bsz x h1

        # userVisAttn: bsz x output_size
        Rating = torch.sum(torch.mul(user_out, item_out), 1, keepdim=True)
        if verbose > 0:
            tqdm.write("\n\tVisRating: {}".format(Rating.size()))  # bsz x 1

        rating_pred = Rating
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include Visual Rating Predict)".format(rating_pred.size()))  # bsz x 1

        # User & Item Bias
        batch_userOffset = self.uid_userOffset(batch_uid)
        batch_itemOffset = self.iid_itemOffset(batch_iid)

        # Include User & Item Doc Bias
        rating_pred = rating_pred + batch_userOffset + batch_itemOffset  # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include User & Item Bias)".format(rating_pred.size()))

        # Include Global Bias
        rating_pred = rating_pred + self.globalOffset  # bsz x 1
        if verbose > 0:
            tqdm.write("rating_pred: {} (Include Global Bias)".format(rating_pred.size()))

        if verbose > 0:
            tqdm.write("\n[VANRA_RatingPred Output] rating_pred: {}".format(rating_pred.size()))
            tqdm.write("**************************************** ***************************** ****************************************\n")

        return rating_pred
