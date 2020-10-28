import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import PAD_idx, UNK_idx

from .MSANR_ARL import MSANR_ARL

from .MSANR_RatingPred import MSANR_RatingPred

from tqdm import tqdm


class MSANR(nn.Module):
    def __init__(self, logger, args, num_users, num_items):
        super(MSANR, self).__init__()

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

        # Aspect Representation Learning - Single Aspect-based Attention Network (Shared between User & Item)
        self.MSANR_ARL = MSANR_ARL(logger, args)

        # Rating Predictor using the 'Simplified Model'
        self.MSANR_RatingPred = MSANR_RatingPred(logger, args, self.num_users, self.num_items)

    def forward(self, batch_uid, batch_iid, verbose=0):
        # Input
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

        # userAspAttn:  bsz x num_aspects x max_doc_len
        # userAspDoc:   bsz x num_aspects x h1
        userAspAttn, userAspDoc = self.MSANR_ARL(batch_userDocEmbed, verbose=verbose)
        itemAspAttn, itemAspDoc = self.MSANR_ARL(batch_itemDocEmbed, verbose=verbose)

        rating_pred = self.MSANR_RatingPred(userAspDoc, itemAspDoc, batch_uid, batch_iid, verbose=verbose)

        return rating_pred
