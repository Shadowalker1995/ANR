import torch
import torch.nn as nn
import torch.nn.functional as F

from .DeepCoNN_RatingPred import DeepCoNN_RatingPred

from tqdm import tqdm


class DeepCoNN(nn.Module):
    """
    2017. Joint Deep Modeling of Users and Items Using Reviews for Recommendation
    """
    def __init__(self, logger, args, num_users, num_items):
        super(DeepCoNN, self).__init__()

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

        # bsz x 1 x max_doc_len x word_embed_dim -> bsz x filters_num x (max_doc_len - ctx_win_size + 1) x 1
        self.user_CNN = nn.Conv2d(1, self.args.filters_num, (self.args.ctx_win_size, self.args.word_embed_dim))
        self.item_CNN = nn.Conv2d(1, self.args.filters_num, (self.args.ctx_win_size, self.args.word_embed_dim))

        self.user_fcLayer = nn.Linear(self.args.filters_num, self.args.output_size)
        self.item_fcLayer = nn.Linear(self.args.filters_num, self.args.output_size)

        if self.args.dropout_rate > 0.0:
            self.userDropout = nn.Dropout(p=self.args.dropout_rate)
            self.itemDropout = nn.Dropout(p=self.args.dropout_rate)

        # Parameter initialization
        for CNN in [self.user_CNN, self.item_CNN]:
            nn.init.xavier_normal_(CNN.weight)
            nn.init.constant_(CNN.bias, 0.1)

        for fcLayer in [self.user_fcLayer, self.item_fcLayer]:
            nn.init.uniform_(fcLayer.weight, -0.1, 0.1)
            nn.init.constant_(fcLayer.bias, 0.1)

        self.DeepCoNN_RatingPred = DeepCoNN_RatingPred(logger, args, self.num_users, self.num_items)

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

        # bsz x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x word_embed_dim
        # bsz x 1 x max_doc_len x word_embed_dim -> bsz x filters_num x (max_doc_len - ctx_win_size + 1) x 1
        # bsz x filters_num x (max_doc_len - ctx_win_size + 1) x 1 -> bsz x filters_num x (max_doc_len - ctx_win_size + 1)
        batch_userFea = F.relu(self.user_CNN(batch_userDocEmbed.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        batch_itemFea = F.relu(self.item_CNN(batch_itemDocEmbed.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)

        # bsz x filters_num x (max_doc_len - ctx_win_size + 1) -> bsz x filters_num x 1
        # bsz x filters_num x 1 -> bsz x filters_num
        batch_userFea = F.max_pool1d(batch_userFea, batch_userFea.size(2)).squeeze(2)
        batch_itemFea = F.max_pool1d(batch_itemFea, batch_itemFea.size(2)).squeeze(2)

        # bsz x filters_num -> bsz x output_size
        batch_userFea = self.userDropout(self.user_fcLayer(batch_userFea))
        batch_itemFea = self.itemDropout(self.item_fcLayer(batch_itemFea))

        # bsz x output_size -> bsz x 1 x output_size
        # batch_userFea = torch.stack([batch_userFea], 1)
        # batch_itemFea = torch.stack([batch_itemFea], 1)

        # bsz x 1
        rating_pred = self.DeepCoNN_RatingPred(batch_userFea, batch_itemFea, batch_uid, batch_iid, verbose=verbose)

        return rating_pred


