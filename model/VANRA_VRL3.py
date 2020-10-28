import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utilities import to_var

from tqdm import tqdm


class VANRA_VRL(nn.Module):
    def __init__(self, logger, args):
        super(VANRA_VRL, self).__init__()

        self.logger = logger
        self.args = args

        # ========== new ==========
        # Vis Embedding Projection Matrices
        self.wedProj_user = nn.Parameter(torch.Tensor(1, self.args.output_size), requires_grad=True)
        self.wedProj_user.data.uniform_(-0.01, 0.01)

        self.wedProj_item = nn.Parameter(torch.Tensor(1, self.args.output_size), requires_grad=True)
        self.wedProj_item.data.uniform_(-0.01, 0.01)
        # ========== new ==========

        self.localAttentionLayer_user = LocalAttention(self.args)
        self.localAttentionLayer_item = LocalAttention(self.args)

    '''
    [Input]     batch_userVis:  bsz x max_vis_len
    [Input]     batch_itemVis:  bsz x max_vis_len
    [Output]    out_user:       bsz x output_size
    [Output]    out_item:       bsz x output_size
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

        # ========== new ===========
        # (bsz x max_doc_len x 1) x (1 x output_size) -> bsz x max_doc_len x output_size
        batch_userVis = torch.matmul(batch_userVis, self.wedProj_user)
        batch_itemVis = torch.matmul(batch_itemVis, self.wedProj_item)
        # ========== new ===========

        local_user = self.localAttentionLayer_user(batch_userVis)
        local_item = self.localAttentionLayer_item(batch_itemVis)
        if verbose > 0:
            tqdm.write("\nlocal_user: {}".format(local_user.size()))                    # bsz x output_size
            tqdm.write("local_item: {}".format(local_item.size()))                      # bsz x output_size

        return local_user, local_item


class LocalAttention(nn.Module):
    def __init__(self, args):
        super(LocalAttention, self).__init__()

        # self.win_size = args.ctx_win_size
        self.win_size = 1

        self.attention_layer = nn.Sequential(
            # bsz x 1 x max_doc_len x output_size -> bsz x 1 x max_vis_len x 1
            nn.Conv2d(1, 1, kernel_size=(self.win_size, args.output_size), padding=((self.win_size - 1)//2, 0)),
            nn.Softmax(dim=2))

    '''
    [Input]     x:      bsz x max_vis_len x output_size
    [Output]    out:    bsz x output_size
    '''
    def forward(self, x):
        # bsz x max_doc_len x output_size -> bsz x 1 x max_doc_len x output_size
        # bsz x 1 x max_doc_len x output_size -> bsz x 1 x max_doc_len x 1
        # bsz x 1 x max_doc_len x 1 -> bsz x max_doc_len x 1
        scores = self.attention_layer(x.unsqueeze(1)).squeeze(1)

        # (bsz x max_doc_len x output_size) * (bsz x max_doc_len x 1) -> bsz x max_doc_len x output_size
        out = torch.mul(x, scores)
        # bsz x max_doc_len x output_size -> bsz x output_size
        out = torch.sum(out, dim=1)

        return out
