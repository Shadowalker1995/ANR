import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utilities import to_var

from tqdm import tqdm


class LocalAttention(nn.Module):
    def __init__(self, input_size, embed_size, win_size, out_channels):
        super(LocalAttention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.win_size = win_size
        self.out_channels = out_channels

        self.attention_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(self.win_size, self.embed_size)),
            nn.Sigmoid())

        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(1, self.embed_size)),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d((self.input_size, 1)))

    def forward(self, x):
        padding = Variable(torch.zeros(x.size(0), int((self.win_size - 1) / 2), self.embed_size))
        padding = padding.cuda()
        x_pad = torch.cat((padding, x, padding), 1)

        x_pad = x_pad.unsqueeze(1)
        scores = self.attention_layer(x_pad)

        scores = scores.squeeze(1)

        out = torch.mul(x, scores)

        out = out.unsqueeze(1)
        out = self.cnn(out)

        return out


class GlobalAttention(nn.Module):
    def __init__(self, input_size, embed_size, out_channels):
        super(GlobalAttention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.out_channels = out_channels

        self.attention_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(self.input_size, self.embed_size)),
            nn.Sigmoid())

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(2, self.embed_size)),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d((self.input_size - 2 + 1, 1)))

        self.cnn_2 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(3, self.embed_size)),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d((self.input_size - 3 + 1, 1)))

        self.cnn_3 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(4, self.embed_size)),
            # nn.Tanh(),
            nn.ReLU(),
            nn.MaxPool2d((self.input_size - 4 + 1, 1)))

    def forward(self, x):
        x = x.unsqueeze(1)
        score = self.attention_layer(x)
        out = torch.mul(x, score)
        out_1 = self.cnn_1(out)
        out_2 = self.cnn_2(out)
        out_3 = self.cnn_3(out)
        return out_1, out_2, out_3


class VANRA_VRL(nn.Module):
    def __init__(self, logger, args):
        super(VANRA_VRL, self).__init__()

        self.logger = logger
        self.args = args
        self.input_size = self.args.max_vis_len
        # self.embed_size = self.args.word_embed_dim
        self.embed_size = 1
        self.win_size = self.args.ctx_win_size
        self.channels_local = self.args.channels_local
        self.channels_global = self.args.channels_global
        self.fc_input_size = self.channels_local + 3 * self.channels_global
        self.hidden_size = self.args.hidden_size
        self.output_size = self.args.output_size

        self.localAttentionLayer_user = LocalAttention(self.input_size, self.embed_size, self.win_size, self.channels_local)
        self.globalAttentionLayer_user = GlobalAttention(self.input_size, self.embed_size, self.channels_global)
        self.localAttentionLayer_item = LocalAttention(self.input_size, self.embed_size, self.win_size, self.channels_local)
        self.globalAttentionLayer_item = GlobalAttention(self.input_size, self.embed_size, self.channels_global)
        self.fcLayer = nn.Sequential(
            nn.Linear(self.fc_input_size, self.hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    '''
    [Input]     batch_visIn:    bsz x max_vis_len
    [Output]    batch_aspRep:   bsz x output_size
    '''
    def forward(self, batch_userVis, batch_itemVis, verbose=0):
        if verbose > 0:
            tqdm.write(
                "\n============================== Visual Representation Learning (VRL) ==============================")
            tqdm.write("[Input] batch_userVis: {}".format(batch_userVis.size()))        # bsz x max_vis_len
            tqdm.write("[Input] batch_itemVis: {}".format(batch_itemVis.size()))        # bsz x max_vis_len

        batch_userVis = batch_userVis.unsqueeze(2)
        batch_itemVis = batch_itemVis.unsqueeze(2)
        if verbose > 0:
            tqdm.write("\nbatch_userVis: {}".format(batch_userVis.size()))              # bsz x max_vis_len x 1
            tqdm.write("batch_itemVis: {}".format(batch_itemVis.size()))                # bsz x max_vis_len x 1

        local_user = self.localAttentionLayer_user(batch_userVis)
        local_item = self.localAttentionLayer_item(batch_itemVis)
        if verbose > 0:
            tqdm.write("\nlocal_user: {}".format(local_user.size()))                    # bsz x channels_local x 1 x 1
            tqdm.write("local_item: {}".format(local_item.size()))                      # bsz x channels_local x 1 x 1

        global1_user, global2_user, global3_user = self.globalAttentionLayer_user(batch_userVis)
        global1_item, global2_item, global3_item = self.globalAttentionLayer_item(batch_itemVis)
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

        out_user = out_user.view(out_user.size(0), -1)
        out_item = out_item.view(out_item.size(0), -1)
        if verbose > 0:
            tqdm.write("\nout_user: {}".format(out_user.size()))                        # bsz x channels_local
            tqdm.write("out_item: {}".format(out_item.size()))                          # bsz x channels_local

        out_user = self.fcLayer(out_user)
        out_item = self.fcLayer(out_item)
        if verbose > 0:
            tqdm.write("\nout_user: {}".format(out_user.size()))                        # bsz x output_size
            tqdm.write("out_item: {}".format(out_item.size()))                          # bsz x output_size

        # out = torch.sum(torch.mul(out_user, out_item), 1)

        return out_user, out_item
