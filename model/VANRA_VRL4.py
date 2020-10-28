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

        self.fcLayer_user = nn.Sequential(
            nn.Dropout(args.dropout_rate),
            # bsz x max_vis_len -> bsz x output_size
            nn.Linear(args.max_vis_len, args.output_size))

        self.fcLayer_item = nn.Sequential(
            nn.Dropout(args.dropout_rate),
            # bsz x max_vis_len -> bsz x output_size
            nn.Linear(args.max_vis_len, args.output_size))

        # max_vis_len x h1
        self.wedProj_user = nn.Parameter(torch.Tensor(self.args.max_vis_len, self.args.h1), requires_grad=True)
        self.wedProj_item = nn.Parameter(torch.Tensor(self.args.max_vis_len, self.args.h1), requires_grad=True)

        self.wedProj_user.data.uniform_(-0.01, 0.01)
        self.wedProj_item.data.uniform_(-0.01, 0.01)

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

        # user_Fea = self.fcLayer_user(batch_userVis)
        # item_Fea = self.fcLayer_item(batch_itemVis)

        user_Fea = torch.matmul(batch_userVis, self.wedProj_user)
        item_Fea = torch.matmul(batch_itemVis, self.wedProj_item)

        if verbose > 0:
            tqdm.write("\nlocal_user: {}".format(user_Fea.size()))                    # bsz x output_size
            tqdm.write("local_item: {}".format(item_Fea.size()))                      # bsz x output_size

        return user_Fea, item_Fea
