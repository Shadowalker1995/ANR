import torch
import torch.nn as nn
import torch.nn.functional as F

from .DAttn_RatingPred import DAttn_RatingPred

from tqdm import tqdm


class DAttn(nn.Module):
    """
    2017. Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction
    Rescys
    This implementation is base on https://github.com/ShomyLiu/Neu-Review-Rec
    """
    def __init__(self, logger, args, num_users, num_items):
        super(DAttn, self).__init__()

        self.logger = logger
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        # User Documents & Item Documents (Input)
        self.uid_userDoc = nn.Embedding(self.num_users, self.args.max_doc_len)  # num_users x max_doc_len
        self.uid_userDoc.weight.requires_grad = False

        self.iid_itemDoc = nn.Embedding(self.num_items, self.args.max_doc_len)  # num_items x max_doc_len
        self.iid_itemDoc.weight.requires_grad = False

        # Word Embeddings (Input)
        self.wid_wEmbed = nn.Embedding(self.args.vocab_size, self.args.word_embed_dim)  # vocab_size x word_embed_dim
        self.wid_wEmbed.weight.requires_grad = False

        self.user_net = Net(logger, args)
        self.item_net = Net(logger, args)

        self.DAttn_RatingPred = DAttn_RatingPred(logger, args, num_users, num_items)

    '''
    [Input]     batch_userDoc:  bsz x max_doc_len
    [Input]     batch_itemDoc:  bsz x max_doc_len
    [Output]    rating_pred:    bsz x 1
    '''
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

        # ========== new ===========
        # (bsz x max_doc_len x word_embed_dim) x (word_embed_dim x 1) -> bsz x max_doc_len x 1
        # batch_userDocEmbed = torch.matmul(batch_userDocEmbed, self.wedProj)
        # batch_itemDocEmbed = torch.matmul(batch_itemDocEmbed, self.wedProj)
        # ========== new ===========

        batch_userFea = self.user_net(batch_userDocEmbed)
        batch_itemFea = self.item_net(batch_itemDocEmbed)

        # bsz x 1
        rating_pred = self.DAttn_RatingPred(batch_userFea, batch_itemFea, batch_uid, batch_iid, verbose=verbose)

        return rating_pred

class Net(nn.Module):
    def __init__(self, logger, args):
        super(Net, self).__init__()

        self.args = args

        self.localAttentionLayer = LocalAttention(logger, args)
        self.globalAttentionLayer = GlobalAttention(logger, args)

        self.fc_input_size = args.channels_local + 3 * args.channels_global

        self.fcLayer = nn.Sequential(
            # bsz x (channels_local + 3 * channels_global) -> bsz x (channels_local + 3 * channels_global)
            nn.Linear(self.fc_input_size, self.fc_input_size),
            nn.Dropout(args.dropout_rate),
            nn.ReLU(),
            # bsz x (channels_local + 3 * channels_global) -> bsz x output_size
            nn.Linear(self.fc_input_size, args.output_size),
        )
        self.dropout = nn.Dropout(args.dropout_rate)
        self.reset_para()

    def forward(self, batch_DocEmbed):
        # bsz x max_doc_len x word_embed_dim -> bsz x channels_local
        local_fea = self.localAttentionLayer(batch_DocEmbed)
        # bsz x max_doc_len x word_embed_dim -> [bsz x channels_global]
        global_fea = self.globalAttentionLayer(batch_DocEmbed)
        # bsz x (channels_local + 3 * channels_global)
        cat_fea = torch.cat([local_fea]+global_fea, 1)
        cat_fea = self.dropout(cat_fea)
        # bsz x (channels_local + 3 * channels_global) -> bsz x output_size
        cat_fea = self.fcLayer(cat_fea)
        return cat_fea

    def reset_para(self):
        cnns = [self.localAttentionLayer.cnn[0], self.localAttentionLayer.attention_layer[0]]
        for cnn in cnns:
            nn.init.xavier_uniform_(cnn.weight, gain=1)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        for cnn in self.globalAttentionLayer.convs:
            nn.init.xavier_uniform_(cnn.weight, gain=1)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        nn.init.uniform_(self.fcLayer[0].weight, -0.1, 0.1)
        nn.init.uniform_(self.fcLayer[-1].weight, -0.1, 0.1)


class LocalAttention(nn.Module):
    def __init__(self, logger, args):
        super(LocalAttention, self).__init__()

        self.win_size = args.ctx_win_size

        self.attention_layer = nn.Sequential(
            # bsz x 1 x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x 1
            nn.Conv2d(1, 1, kernel_size=(self.win_size, args.word_embed_dim), padding=((self.win_size - 1)//2, 0)),
            # bsz x 1 x max_doc_len x 1
            nn.Sigmoid(),
            # nn.Softmax(dim=2),
        )

        self.cnn = nn.Sequential(
            # bsz x 1 x max_doc_len x word_embed_dim -> bsz x channels_local x max_doc_len x 1
            nn.Conv2d(1, args.channels_local, kernel_size=(1, args.word_embed_dim)),
            nn.Tanh(),
        )

    '''
    [Input]     x:      bsz x max_doc_len x word_embed_dim
    [Output]    out:    bsz x channels_local
    '''
    def forward(self, x):
        # bsz x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x word_embed_dim
        # bsz x 1 x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x 1
        # bsz x 1 x max_doc_len x 1 -> bsz x max_doc_len x 1
        scores = self.attention_layer(x.unsqueeze(1)).squeeze(1)
        # (bsz x max_doc_len x word_embed_dim) * (bsz x max_doc_len x 1) -> bsz x max_doc_len x word_embed_dim
        # bsz x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x word_embed_dim
        out = torch.mul(x, scores).unsqueeze(1)
        # bsz x 1 x max_doc_len x word_embed_dim -> bsz x channels_local x max_doc_len x 1
        # bsz x channels_local x max_doc_len x 1 -> bsz x channels_local x max_doc_len
        out = self.cnn(out).squeeze(3)
        # bsz x channels_local x max_doc_len -> bsz x channels_local x 1
        # bsz x channels_local x 1 -> bsz x channels_local
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return out


class GlobalAttention(nn.Module):
    def __init__(self, logger, args, filters_size=None):
        super(GlobalAttention, self).__init__()

        if filters_size is None:
            filters_size = [2, 3, 4]

        self.attention_layer = nn.Sequential(
            # bsz x 1 x max_doc_len x word_embed_dim -> bsz x 1 x 1 x 1
            nn.Conv2d(1, 1, kernel_size=(args.max_doc_len, args.word_embed_dim)),
            # bsz x 1 x 1 x 1
            nn.Sigmoid()
        )

        # bsz x 1 x max_doc_len x word_embed_dim -> bsz x channels_global x (max_doc_len-k+1) x 1
        self.convs = nn.ModuleList([nn.Conv2d(1, args.channels_global, (k, args.word_embed_dim)) for k in filters_size])

    '''
    [Input]     x:          bsz x max_doc_len x word_embed_dim
    [Output]    conv_outs:  [bsz x channels_global]
    '''
    def forward(self, x):
        # bsz x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x word_embed_dim
        x = x.unsqueeze(1)
        # bsz x 1 x max_doc_len x word_embed_dim -> bsz x 1 x 1 x 1
        score = self.attention_layer(x)
        # (bsz x 1 x max_doc_len x word_embed_dim) * (bsz x 1 x 1 x 1) -> bsz x 1 x max_doc_len x word_embed_dim
        x = torch.mul(x, score)
        # [bsz x 1 x max_doc_len x word_embed_dim -> bsz x channels_global x (max_doc_len-k+1) x 1]
        # [bsz x channels_global x (max_doc_len-k+1) x 1 -> bsz x channels_global x (max_doc_len-k+1)]
        conv_outs = [torch.tanh(cnn(x).squeeze(3)) for cnn in self.convs]
        # [bsz x channels_global x (max_doc_len-k+1) -> bsz x channels_global x 1]
        # [bsz x channels_global x 1 -> bsz x channels_global]
        conv_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]
        return conv_outs
