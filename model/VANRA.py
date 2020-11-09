import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import PAD_idx, UNK_idx

from .MSANR_ARL import MSANR_ARL
from .VANRA_VRL4 import VANRA_VRL

# from .VANRA_RatingPred import VANRA_RatingPred
from .VANRA_RatingPred_fc import VANRA_RatingPred

from tqdm import tqdm


class VANRA(nn.Module):
    """
    This is the complete Visual & Aspect-based Neural Recommender with Attention Mechanisim (VANRA), with ARL, AIE and VRL as its main components.
    """
    def __init__(self, logger, args, num_users, num_items):
        super(VANRA, self).__init__()

        self.logger = logger
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        # User Documents & Item Documents (Input)
        self.uid_userDoc = nn.Embedding(self.num_users, self.args.max_doc_len)              # num_users x max_doc_len
        self.uid_userDoc.weight.requires_grad = False

        self.iid_itemDoc = nn.Embedding(self.num_items, self.args.max_doc_len)              # num_items x max_doc_len
        self.iid_itemDoc.weight.requires_grad = False

        # Word Embeddings (Input)
        self.wid_wEmbed = nn.Embedding(self.args.vocab_size, self.args.word_embed_dim)      # vocab_size x word_embed_dim
        self.wid_wEmbed.weight.requires_grad = False

        # User Visual Feature & Item Visual Feature (Input)
        self.uid_userVis = nn.Embedding(self.num_users, self.args.max_vis_len)              # num_users x max_vis_len
        self.uid_userVis.weight.requires_grad = False

        self.iid_itemVis = nn.Embedding(self.num_items, self.args.max_vis_len)              # num_users x max_vis_len
        self.iid_itemVis.weight.requires_grad = False

        # Aspect Representation Learning - Single Aspect-based Attention Network (Shared between User & Item)
        self.MSANR_ARL = MSANR_ARL(logger, args)

        # Visual Representation Learning - Single Visual-based Attention Network (Shared between User & Item)
        self.VANRA_VRL = VANRA_VRL(logger, args)

        # Visual & Aspect Based Rating Predictor with Attention Mechanism
        self.VANRA_RatingPred = VANRA_RatingPred(logger, args, self.num_users, self.num_items)

    def forward(self, batch_uid, batch_iid, verbose=0):
        # Input - Document Date
        batch_userDoc = self.uid_userDoc(batch_uid)
        batch_itemDoc = self.iid_itemDoc(batch_iid)
        if verbose > 0:
            tqdm.write("batch_userDoc: {}".format(batch_userDoc.size()))                # bsz x max_doc_len
            tqdm.write("batch_itemDoc: {}".format(batch_itemDoc.size()))                # bsz x max_doc_len

        # Embedding Layer
        batch_userDocEmbed = self.wid_wEmbed(batch_userDoc.long())
        batch_itemDocEmbed = self.wid_wEmbed(batch_itemDoc.long())
        if verbose > 0:
            tqdm.write("batch_userDocEmbed: {}".format(batch_userDocEmbed.size()))      # bsz x max_doc_len x word_embed_dim
            tqdm.write("batch_itemDocEmbed: {}".format(batch_itemDocEmbed.size()))      # bsz x max_doc_len x word_embed_dim

        # Input - Visual Feature Data
        batch_userVis = self.uid_userVis(batch_uid)
        batch_itemVis = self.iid_itemVis(batch_iid)
        if verbose > 0:
            tqdm.write("batch_userVis: {}".format(batch_userVis.size()))                # bsz x max_vis_len
            tqdm.write("batch_itemVis: {}".format(batch_itemVis.size()))                # bsz x max_vis_len

        # userAspAttn:  bsz x num_aspects x max_doc_len
        # userAspDoc:   bsz x num_aspects x h1
        userAspAttn, userAspDoc = self.MSANR_ARL(batch_userDocEmbed, verbose=verbose)
        itemAspAttn, itemAspDoc = self.MSANR_ARL(batch_itemDocEmbed, verbose=verbose)

        userVis, itemVis = self.VANRA_VRL(batch_userVis, batch_itemVis, verbose=verbose)

        rating_pred = self.VANRA_RatingPred(userAspDoc, itemAspDoc, userVis, itemVis, batch_uid, batch_iid, verbose=verbose)

        if verbose > 0:
            # bsz x 1
            tqdm.write("\n[Final Output of {}] rating_pred: {}\n".format(self.args.model, rating_pred.size()))

        return rating_pred
