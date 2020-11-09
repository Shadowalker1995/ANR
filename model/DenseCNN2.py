import torch
import torch.nn as nn
import torch.nn.functional as F

from .DenseCNN_RatingPred import DenseCNN_RatingPred

from tqdm import tqdm


class DenseCNN(nn.Module):
    """
    2018. Densely Connected CNN with Multi-scale Feature Attention for Text Classification
    NOT DENSE
    with token attention mechanism in each convolution layer and
    scale attention mechanism in the end.
    """
    def __init__(self, logger, args, num_users, num_items):
        super(DenseCNN, self).__init__()

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

        # Word Embedding Projection Matrices
        self.docProj = nn.Parameter(torch.Tensor(self.args.word_embed_dim, self.args.output_size), requires_grad=True)
        self.docProj.data.uniform_(-0.01, 0.01)

        self.doc_net = DenseNet(logger, args)

        self.DenseCNN_RatingPred = DenseCNN_RatingPred(logger, args, num_users, num_items)

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

        # (bsz x max_doc_len x word_embed_dim) x (word_embed_dim x output_size) -> bsz x max_doc_len x output_size
        batch_userDocEmbed = torch.matmul(batch_userDocEmbed, self.docProj)
        batch_itemDocEmbed = torch.matmul(batch_itemDocEmbed, self.docProj)

        batch_userFea = self.doc_net(batch_userDocEmbed)
        batch_itemFea = self.doc_net(batch_itemDocEmbed)

        # bsz x 1
        rating_pred = self.DenseCNN_RatingPred(batch_userFea, batch_itemFea, batch_uid, batch_iid, verbose=verbose)

        return rating_pred


class DenseNet(nn.Module):
    def __init__(self, logger, args, filters_size=None):
        super(DenseNet, self).__init__()

        self.args = args

        if filters_size is None:
            filters_size = [3, 5, 7, 9]
            # filters_size = [3]

        self.local_attention = nn.ModuleList([nn.Sequential(
            # bsz x 1 x max_doc_len x output_size -> bsz x 1 x max_doc_len x 1
            nn.Conv2d(1, 1, kernel_size=(k, args.output_size), padding=((k - 1)//2, 0)),
            # bsz x 1 x max_doc_len x 1
            nn.Softmax(dim=2),
        ) for k in filters_size])

        # bsz x 1 x filters_num x output_size -> bsz x 1 x filters_num x 1
        self.scale_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, args.output_size)),
            nn.Softmax(dim=2))

        self.dropout = nn.Dropout(args.dropout_rate)

    def reset_para(self):
        for cnn in self.local_attention:
            nn.init.xavier_uniform_(cnn[0].weight, gain=1)
            nn.init.uniform_(cnn[0].bias, -0.1, 0.1)
        nn.init.xavier_uniform_(self.global_attention[0].weight, gain=1)
        nn.init.uniform_(self.global_attention[0].bias, -0.1, 0.1)
        nn.init.xavier_uniform_(self.scale_attention[0].weight, gain=1)
        nn.init.uniform_(self.scale_attention[0].bias, -0.1, 0.1)

    '''
    [Input]     x:      bsz x max_doc_len x output_size
    [Output]    out:    bsz x output_size
    '''
    def forward(self, batch_DocEmbed):
        # [bsz x max_doc_len x output_size] -> [bsz x 1 x max_doc_len x output_size]
        # [bsz x 1 x max_doc_len x output_size] -> [bsz x 1 x max_doc_len x 1]
        # [bsz x 1 x max_doc_len x 1] -> [bsz x max_doc_len x 1]
        scores = [attention_layer(batch_DocEmbed.unsqueeze(1)).squeeze(1) for attention_layer in self.local_attention]

        # [(bsz x max_doc_len x output_size) * (bsz x max_doc_len x 1)] -> [bsz x max_doc_len x output_size]
        # [bsz x max_doc_len x output_size] -> [bsz x output_size]
        local_feas = [torch.sum(torch.mul(batch_DocEmbed, score), dim=1) for score in scores]

        local_feas = [self.dropout(local_fea) for local_fea in local_feas]

        # bsz x filters_num x output_size
        # feas = torch.stack(local_feas+[global_fea], dim=1)
        feas = torch.stack(local_feas, dim=1)

        # bsz x filters_num x output_size -> bsz x 1 x filters_num x output_size
        # bsz x 1 x filters_num x output_size -> bsz x 1 x filters_num x 1
        # bsz x 1 x filters_num x 1 -> bsz x filters_num x 1
        scale_score = self.scale_attention(feas.unsqueeze(1)).squeeze(1)
        print(scale_score[0])
        # bsz x filters_num x output_size * bsz x filters_num x 1 -> bsz x filters_num x output_size
        # bsz x filters_num x output_size -> bsz x output_size
        fea = torch.sum(torch.mul(feas, scale_score), dim=1)

        return fea
