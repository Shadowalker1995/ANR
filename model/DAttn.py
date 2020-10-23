import torch
import torch.nn as nn
from torch.autograd import Variable

from .DAttn_RatingPred import DAttn_RatingPred

from tqdm import tqdm


class LocalAttention(nn.Module):
    def __init__(self, input_size, embed_size, win_size, out_channels):
        super(LocalAttention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.win_size = win_size
        self.out_channels = out_channels

        self.attention_layer = nn.Sequential(
            # bsz x 1 x (max_vis_len + (win_size - 1) / 2) x word_embed_dim -> bsz x 1 x max_vis_len x word_embed_dim
            nn.Conv2d(1, 1, kernel_size=(self.win_size, self.embed_size)),
            # bsz x 1 x max_vis_len x word_embed_dim
            # nn.Sigmoid(),
            nn.Softmax(dim=2),
        )

        self.cnn = nn.Sequential(
            # bsz x 1 x max_vis_len x word_embed_dim -> bsz x out_channels x max_vis_len x 1
            nn.Conv2d(1, self.out_channels, kernel_size=(1, self.embed_size)),
            nn.Tanh(),
            # nn.ReLU(),
            # bsz x out_channels x max_vis_len x 1 -> bsz x out_channels x 1 x 1
            nn.MaxPool2d((self.input_size, 1)))

    '''
    [Input]     x:      bsz x max_doc_len x word_embed_dim
    [Output]    out:    bsz x channels_local x 1 x 1
    '''
    def forward(self, x):
        padding = Variable(torch.zeros(x.size(0), int((self.win_size - 1) / 2), self.embed_size))
        padding = padding.cuda()
        x_pad = torch.cat((padding, x, padding), 1)                 # bsz x (max_vis_len + (win_size - 1) / 2) x word_embed_dim
        x_pad = x_pad.unsqueeze(1)                                  # bsz x 1 x (max_vis_len + (win_size - 1) / 2) x word_embed_dim
        scores = self.attention_layer(x_pad)                        # bsz x 1 x max_vis_len x word_embed_dim
        scores = scores.squeeze(1)                                  # bsz x max_vis_len x word_embed_dim

        out = torch.mul(x, scores)                                  # bsz x max_vis_len x word_embed_dim

        out = out.unsqueeze(1)                                      # bsz x 1 x max_vis_len x word_embed_dim
        out = self.cnn(out)                                         # bsz x out_channels x 1 x 1

        return out


class GlobalAttention(nn.Module):
    def __init__(self, input_size, embed_size, out_channels):
        super(GlobalAttention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.out_channels = out_channels

        self.attention_layer = nn.Sequential(
            # bsz x 1 x max_vis_len x word_embed_dim -> bsz x 1 x 1 x 1
            nn.Conv2d(1, 1, kernel_size=(self.input_size, self.embed_size)),
            # bsz x 1 x 1 x 1
            nn.Sigmoid())

        self.cnn_1 = nn.Sequential(
            # bsz x 1 x max_vis_len x word_embed_dim -> bsz x out_channels x (max_vis_len - 2 + 1) x 1
            nn.Conv2d(1, self.out_channels, kernel_size=(2, self.embed_size)),
            nn.Tanh(),
            # bsz x out_channels x (max_vis_len - 2 + 1) x 1 -> bsz x out_channels x 1 x 1
            nn.MaxPool2d((self.input_size - 2 + 1, 1)))

        self.cnn_2 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(3, self.embed_size)),
            nn.Tanh(),
            nn.MaxPool2d((self.input_size - 3 + 1, 1)))

        self.cnn_3 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(4, self.embed_size)),
            nn.Tanh(),
            nn.MaxPool2d((self.input_size - 4 + 1, 1)))

    '''
    [Input]     x:          bsz x max_doc_len x word_embed_dim
    [Output]    out1/2/3:   bsz x channels_global x 1 x 1
    '''
    def forward(self, x):
        x = x.unsqueeze(1)                                          # bsz x 1 x max_vis_len x word_embed_dim
        score = self.attention_layer(x)                             # bsz x 1 x 1 x 1
        out = torch.mul(x, score)                                   # bsz x 1 x max_vis_len x word_embed_dim
        out_1 = self.cnn_1(out)                                     # bsz x channels_global x 1 x 1
        out_2 = self.cnn_2(out)                                     # bsz x channels_global x 1 x 1
        out_3 = self.cnn_3(out)                                     # bsz x channels_global x 1 x 1
        return out_1, out_2, out_3


class DAttn(nn.Module):
    def __init__(self, logger, args, num_users, num_items):
        super(DAttn, self).__init__()

        self.logger = logger
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        self.input_size = self.args.max_vis_len
        # self.embed_size = self.args.word_embed_dim
        self.embed_size = 1
        self.win_size = self.args.ctx_win_size
        self.channels_local = self.args.channels_local
        self.channels_global = self.args.channels_global
        self.fc_input_size = self.channels_local + 3 * self.channels_global
        self.hidden_size = self.args.hidden_size
        self.output_size = self.args.output_size

        # User Documents & Item Documents (Input)
        self.uid_userDoc = nn.Embedding(self.num_users, self.args.max_doc_len)              # num_users x max_doc_len
        self.uid_userDoc.weight.requires_grad = False

        self.iid_itemDoc = nn.Embedding(self.num_items, self.args.max_doc_len)              # num_items x max_doc_len
        self.iid_itemDoc.weight.requires_grad = False

        # Word Embeddings (Input)
        self.wid_wEmbed = nn.Embedding(self.args.vocab_size, self.args.word_embed_dim)      # vocab_size x word_embed_dim
        self.wid_wEmbed.weight.requires_grad = False

        # ========== new ==========
        # Word Embedding Projection Matrices
        self.wedProj = nn.Parameter(torch.Tensor(self.args.word_embed_dim, 1), requires_grad=True)
        # ========== new ==========

        self.DAttn_RatingPred = DAttn_RatingPred(logger, args, self.num_users, self.num_items)

        self.localAttentionLayer_user = LocalAttention(self.input_size, self.embed_size, self.win_size, self.channels_local)
        self.globalAttentionLayer_user = GlobalAttention(self.input_size, self.embed_size, self.channels_global)
        self.localAttentionLayer_item = LocalAttention(self.input_size, self.embed_size, self.win_size, self.channels_local)
        self.globalAttentionLayer_item = GlobalAttention(self.input_size, self.embed_size, self.channels_global)
        self.fcLayer = nn.Sequential(
            # bsz x fc_input_size -> bsz x hidden_size
            nn.Linear(self.fc_input_size, self.hidden_size),
            nn.Dropout(self.args.dropout_rate),
            nn.ReLU(),
            # bsz x hidden_size -> bsz x output_size
            nn.Linear(self.hidden_size, self.output_size),
        )

    '''
    [Input]     batch_userDoc:  bsz x max_vis_len
    [Input]     batch_itemDoc:  bsz x max_vis_len
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
        batch_userDocEmbed = torch.matmul(batch_userDocEmbed, self.wedProj)
        batch_itemDocEmbed = torch.matmul(batch_itemDocEmbed, self.wedProj)
        # ========== new ===========

        local_user = self.localAttentionLayer_user(batch_userDocEmbed)
        local_item = self.localAttentionLayer_item(batch_itemDocEmbed)
        if verbose > 0:
            tqdm.write("\nlocal_user: {}".format(local_user.size()))                    # bsz x channels_local x 1 x 1
            tqdm.write("local_item: {}".format(local_item.size()))                      # bsz x channels_local x 1 x 1

        global1_user, global2_user, global3_user = self.globalAttentionLayer_user(batch_userDocEmbed)
        global1_item, global2_item, global3_item = self.globalAttentionLayer_item(batch_itemDocEmbed)
        if verbose > 0:
            tqdm.write("\nglobal1_user: {}".format(global1_user.size()))                # bsz x channels_global x 1 x 1
            tqdm.write("global2_user: {}".format(global2_user.size()))                  # bsz x channels_global x 1 x 1
            tqdm.write("global3_user: {}".format(global3_user.size()))                  # bsz x channels_global x 1 x 1
            tqdm.write("global1_item: {}".format(global1_item.size()))                  # bsz x channels_global x 1 x 1
            tqdm.write("global2_item: {}".format(global2_item.size()))                  # bsz x channels_global x 1 x 1
            tqdm.write("global3_item: {}".format(global1_item.size()))                  # bsz x channels_global x 1 x 1

        out_user = torch.cat((local_user, global1_user, global2_user, global3_user), 1)
        out_item = torch.cat((local_item, global1_item, global2_item, global3_item), 1)
        if verbose > 0:
            tqdm.write("\nout_user: {}".format(out_user.size()))                        # bsz x fc_input_size x 1 x 1
            tqdm.write("out_item: {}".format(out_item.size()))                          # bsz x fc_input_size x 1 x 1

        out_user = out_user.contiguous().view(out_user.size(0), -1)
        out_item = out_item.contiguous().view(out_item.size(0), -1)
        if verbose > 0:
            tqdm.write("\nout_user: {}".format(out_user.size()))                        # bsz x fc_input_size
            tqdm.write("out_item: {}".format(out_item.size()))                          # bsz x fc_input_size

        out_user = self.fcLayer(out_user)
        out_item = self.fcLayer(out_item)
        if verbose > 0:
            tqdm.write("\nout_user: {}".format(out_user.size()))                        # bsz x output_size
            tqdm.write("out_item: {}".format(out_item.size()))                          # bsz x output_size

        # rating_pred = torch.sum(torch.mul(out_user, out_item), 1)                       # bsz x 1
        # bsz x 1
        rating_pred = self.DAttn_RatingPred(out_user, out_item, batch_uid, batch_iid, verbose=verbose)

        return rating_pred
