import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utilities import to_var

from .DAttn_RatingPred import DAttn_RatingPred

from tqdm import tqdm


class LocalAttention(nn.Module):
    def __init__(self, args, input_size, embed_size, win_size, out_channels, hidden_size, output_size):
        super(LocalAttention, self).__init__()

        self.args = args
        self.input_size = input_size
        self.embed_size = embed_size
        self.win_size = win_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.output_size = output_size

        # win_size = 3
        self.attention_layer1 = nn.Sequential(
            # bsz x 1 x (max_vis_len + (win_size - 1) / 2) x word_embed_dim(1) -> bsz x 1 x max_vis_len x word_embed_dim(1)
            nn.Conv2d(1, 1, kernel_size=(self.win_size, self.embed_size)),
            # bsz x 1 x max_vis_len x word_embed_dim(1)
            nn.Softmax(dim=2),
        )

        # win_size = 5
        self.attention_layer2 = nn.Sequential(
            # bsz x 1 x (max_vis_len + (win_size - 1) / 2) x word_embed_dim(1) -> bsz x 1 x max_vis_len x word_embed_dim(1)
            nn.Conv2d(1, 1, kernel_size=(self.win_size+2, self.embed_size)),
            # bsz x 1 x max_vis_len x word_embed_dim(1)
            nn.Softmax(dim=2),
        )

        # win_size = 7
        self.attention_layer3 = nn.Sequential(
            # bsz x 1 x (max_vis_len + (win_size - 1) / 2) x word_embed_dim(1) -> bsz x 1 x max_vis_len x word_embed_dim(1)
            nn.Conv2d(1, 1, kernel_size=(self.win_size+4, self.embed_size)),
            # bsz x 1 x max_vis_len x word_embed_dim(1)
            nn.Softmax(dim=2),
        )

    '''
    [Input]     x:      bsz x max_doc_len x word_embed_dim(1)
    [Output]    out:    bsz x channels_local x 1 x 1
    '''
    def forward(self, x):
        padding = Variable(torch.zeros(x.size(0), int((self.win_size - 1) / 2), self.embed_size))
        padding = padding.cuda()

        padding1 = padding
        padding2 = torch.cat((padding, padding), 1)
        padding3 = torch.cat((padding, padding, padding), 1)

        x_pad1 = torch.cat((padding1, x, padding1), 1)                 # bsz x (max_vis_len + (win_size - 1)) x word_embed_dim(1)
        x_pad1 = x_pad1.unsqueeze(1)                                  # bsz x 1 x (max_vis_len + (win_size - 1)) x word_embed_dim(1)

        x_pad2 = torch.cat((padding2, x, padding2), 1)  # bsz x (max_vis_len + (win_size - 1)) x word_embed_dim(1)
        x_pad2 = x_pad2.unsqueeze(1)  # bsz x 1 x (max_vis_len + (win_size - 1)) x word_embed_dim(1)

        x_pad3 = torch.cat((padding3, x, padding3), 1)  # bsz x (max_vis_len + (win_size - 1)) x word_embed_dim(1)
        x_pad3 = x_pad3.unsqueeze(1)  # bsz x 1 x (max_vis_len + (win_size - 1)) x word_embed_dim(1)

        scores1 = self.attention_layer1(x_pad1)                     # bsz x 1 x max_vis_len x word_embed_dim(1)
        scores1 = scores1.squeeze(1)                                # bsz x max_vis_len x word_embed_dim(1)

        scores2 = self.attention_layer2(x_pad2)                     # bsz x 1 x max_vis_len x word_embed_dim(1)
        scores2 = scores2.squeeze(1)                                # bsz x max_vis_len x word_embed_dim(1)

        scores3 = self.attention_layer3(x_pad3)                     # bsz x 1 x max_vis_len x word_embed_dim(1)
        scores3 = scores3.squeeze(1)                                # bsz x max_vis_len x word_embed_dim(1)

        out1 = torch.mul(x, scores1)                                # bsz x max_vis_len x word_embed_dim(1)
        out1 = out1.squeeze(2)                                      # bsz x max_vis_len

        out2 = torch.mul(x, scores2)                                # bsz x max_vis_len x word_embed_dim(1)
        out2 = out2.squeeze(2)                                      # bsz x max_vis_len

        out3 = torch.mul(x, scores3)                                # bsz x max_vis_len x word_embed_dim(1)
        out3 = out3.squeeze(2)                                      # bsz x max_vis_len

        return out1, out2, out3


class GlobalAttention(nn.Module):
    def __init__(self, args, input_size, embed_size, out_channels, hidden_size, output_size):
        super(GlobalAttention, self).__init__()

        self.args = args
        self.input_size = input_size
        self.embed_size = embed_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attention_layer = nn.Sequential(
            # bsz x 1 x max_vis_len x word_embed_dim -> bsz x 1 x 1 x 1
            nn.Conv2d(1, 1, kernel_size=(self.input_size, self.embed_size)),
            # bsz x 1 x 1 x 1
            nn.Sigmoid())

    '''
    [Input]     x:          bsz x max_doc_len x word_embed_dim
    [Output]    out1/2/3:   bsz x channels_global
    '''
    def forward(self, x):
        x = x.unsqueeze(1)                                          # bsz x 1 x max_vis_len x 1
        score = self.attention_layer(x)                             # bsz x 1 x 1 x 1
        out = torch.mul(x, score)                                   # bsz x 1 x max_vis_len x 1
        out = out.squeeze(3)                                         # bsz x max_vis_len
        out = out.squeeze(1)  # bsz x max_vis_len

        return out


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
        self.fc_input_size = 4 * self.args.max_vis_len
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
        self.wedProj.data.uniform_(-0.01, 0.01)
        # ========== new ==========

        self.DAttn_RatingPred = DAttn_RatingPred(logger, args, self.num_users, self.num_items)

        self.localAttentionLayer_user = LocalAttention(self.args, self.input_size, self.embed_size, self.win_size,
                                                       self.channels_local, self.hidden_size, self.output_size)
        self.globalAttentionLayer_user = GlobalAttention(self.args, self.input_size, self.embed_size,
                                                         self.channels_global, self.hidden_size, self.output_size)
        self.localAttentionLayer_item = LocalAttention(self.args, self.input_size, self.embed_size, self.win_size,
                                                       self.channels_local, self.hidden_size, self.output_size)
        self.globalAttentionLayer_item = GlobalAttention(self.args, self.input_size, self.embed_size,
                                                         self.channels_global, self.hidden_size, self.output_size)
        self.fcLayer = nn.Sequential(
            # bsz x fc_input_size -> bsz x hidden_size
            nn.Linear(self.fc_input_size, self.hidden_size),
            # nn.Dropout(self.args.dropout_rate),
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

        local1_user, local2_user, local3_user = self.localAttentionLayer_user(batch_userDocEmbed)
        local1_item, local2_item, local3_item = self.localAttentionLayer_item(batch_itemDocEmbed)
        if verbose > 0:
            tqdm.write("\nlocal1_user: {}".format(local1_user.size()))                  # bsz x output_size
            tqdm.write("local2_user: {}".format(local2_user.size()))                    # bsz x output_size
            tqdm.write("local3_user: {}".format(local3_user.size()))                    # bsz x output_size
            tqdm.write("local1_item: {}".format(local1_item.size()))                    # bsz x output_size
            tqdm.write("local2_item: {}".format(local2_item.size()))                    # bsz x output_size
            tqdm.write("local3_item: {}".format(local3_item.size()))                    # bsz x output_size

        global_user = self.globalAttentionLayer_user(batch_userDocEmbed)
        global_item = self.globalAttentionLayer_item(batch_itemDocEmbed)
        if verbose > 0:
            tqdm.write("\nglobal_user: {}".format(global_user.size()))                  # bsz x output_size
            tqdm.write("global_item: {}".format(global_item.size()))                    # bsz x output_size

        out_user = torch.cat((local1_user, local2_user, local3_user, global_user), 1)
        out_item = torch.cat((local1_item, local2_item, local3_item, global_item), 1)
        if verbose > 0:
            tqdm.write("\nout_user: {}".format(out_user.size()))                        # bsz x (4 x output_size)
            tqdm.write("out_item: {}".format(out_item.size()))                          # bsz x (4 x output_size)

        out_user = self.fcLayer(out_user)
        out_item = self.fcLayer(out_item)
        if verbose > 0:
            tqdm.write("\nout_user: {}".format(out_user.size()))                        # bsz x output_size
            tqdm.write("out_item: {}".format(out_item.size()))                          # bsz x output_size

        # bsz x 1
        rating_pred = self.DAttn_RatingPred(out_user, out_item, batch_uid, batch_iid, verbose=verbose)

        return rating_pred
