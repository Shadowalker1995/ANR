import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class ANR_AIE(nn.Module):
    """
    Aspect Importance Estimation (AIE)
    """

    def __init__(self, logger, args):
        super(ANR_AIE, self).__init__()

        self.logger = logger
        self.args = args

        # Matrix for Interaction between User Aspect-level Representations & Item Aspect-level Representations 
        # This is a learnable (h1 x h1) matrix, i.e. User Aspects - Rows, Item Aspects - Columns
        self.W_a = nn.Parameter(torch.Tensor(self.args.h1, self.args.h1), requires_grad=True)   # W_s: h1 x h1

        # User "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_u = nn.Parameter(torch.Tensor(self.args.h1, self.args.h2), requires_grad=True)   # W_x: h1 x h2
        self.w_hu = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad=True)             # v_x: h2 x 1

        # Item "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_i = nn.Parameter(torch.Tensor(self.args.h1, self.args.h2), requires_grad=True)   # W_y: h1 x h2
        self.w_hi = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad=True)             # v_y: h2 x 1

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.W_a.data.uniform_(-0.01, 0.01)

        self.W_u.data.uniform_(-0.01, 0.01)
        self.w_hu.data.uniform_(-0.01, 0.01)

        self.W_i.data.uniform_(-0.01, 0.01)
        self.w_hi.data.uniform_(-0.01, 0.01)

    '''
    [Input]  userAspRep: bsz x num_aspects x h1 P_u
    [Input]  itemAspRep: bsz x num_aspects x h1 Q_i
    '''
    def forward(self, userAspRep, itemAspRep, verbose=0):
        if verbose > 0:
            tqdm.write(
                "\n\n============================== Aspect Importance Estimation (AIE) ==============================")
            tqdm.write("[Input to AIE] userAspRep: {}".format(userAspRep.size()))   # bsz x num_aspects x h1, P_u
            tqdm.write("[Input to AIE] itemAspRep: {}".format(itemAspRep.size()))   # bsz x num_aspects x h1, Q_i

        userAspRepTrans = torch.transpose(userAspRep, 1, 2)                         # bsz x h1 x num_aspects
        itemAspRepTrans = torch.transpose(itemAspRep, 1, 2)                         # bsz x h1 x num_aspects
        if verbose > 0:
            tqdm.write("\nuserAspRepTrans: {}".format(userAspRepTrans.size()))
            tqdm.write("itemAspRepTrans: {}".format(itemAspRepTrans.size()))

        '''
        Affinity Matrix (User Aspects x Item Aspects), i.e. User Aspects - Rows, Item Aspects - Columns
        S = RELU(P_u * W_s * Q_i^T)
        '''
        # (bsz x num_aspects x h1) * (h1 x h1) -> bsz x num_aspects x h1
        affinityMatrix = torch.matmul(userAspRep, self.W_a)                         # bsz x num_aspects x h1
        if verbose > 0:
            tqdm.write("\naffinityMatrix: {}".format(affinityMatrix.size()))

        # (bsz x num_aspects x h1) * (bsz x h1 x num_aspects) -> bsz x num_aspects x num_aspects
        affinityMatrix = torch.matmul(affinityMatrix, itemAspRepTrans)              # bsz x num_aspects x num_aspects
        if verbose > 0:
            tqdm.write("affinityMatrix: {}".format(affinityMatrix.size()))

        # Non-Linearity: ReLU
        affinityMatrix = F.relu(affinityMatrix)                                     # bsz x num_aspects x num_aspects

        '''
        H_u = RELU(P_u * W_x + S^T * (Q_i * W_y))
        beta_u = softmax(H_u * v_x)
        '''
        # =========== User Importance (over Aspects) ===========
        # (bsz x num_aspects x h1) * (h1 x h2) -> bsz x num_aspects x h2
        H_u_1 = torch.matmul(userAspRep, self.W_u)                                  # bsz x num_aspects x h2
        H_u_2 = torch.matmul(itemAspRep, self.W_i)                                  # bsz x num_aspects x h2

        # (bsz x num_aspects x num_aspects) * (bsz x num_aspects x h2) -> bsz x num_aspects x h2
        H_u_2 = torch.matmul(torch.transpose(affinityMatrix, 1, 2), H_u_2)          # bsz x num_aspects x h2
        H_u = H_u_1 + H_u_2                                                         # bsz x num_aspects x h2

        # Non-Linearity: ReLU
        H_u = F.relu(H_u)                                                           # bsz x num_aspects x h2

        # User Aspect-level Importance
        # (bsz x num_aspects x h2) * (h2 x 1) -> bsz x num_aspects x 1
        userAspImpt = torch.matmul(H_u, self.w_hu)                                  # bsz x num_aspects x 1
        if verbose > 0:
            tqdm.write("\nuserAspImpt: {}".format(userAspImpt.size()))

        userAspImpt = F.softmax(userAspImpt, dim=1)                                 # bsz x num_aspects x 1
        if verbose > 0:
            tqdm.write("userAspImpt: {}".format(userAspImpt.size()))

        userAspImpt = torch.squeeze(userAspImpt, 2)                                 # bsz x num_aspects, beta_u
        if verbose > 0:
            tqdm.write("userAspImpt: {}".format(userAspImpt.size()))
        # =========== User Importance (over Aspects) ===========

        '''
        H_i = RELU(Q_i * W_y + S * (P_u * W_x))
        beta_i = softmax(H_i * v_y)
        '''
        # =========== Item Importance (over Aspects) ===========
        # (bsz x num_aspects x h1) * (h1 x h2) -> bsz x num_aspects x h2
        H_i_1 = torch.matmul(itemAspRep, self.W_i)                                  # bsz x num_aspects x h2
        H_i_2 = torch.matmul(userAspRep, self.W_u)                                  # bsz x num_aspects x h2

        # (bsz x num_aspects x num_aspects) * (bsz x num_aspects x h2) -> bsz x num_aspects x h2
        H_i_2 = torch.matmul(affinityMatrix, H_i_2)                                 # bsz x num_aspects x h2
        H_i = H_i_1 + H_i_2                                                         # bsz x num_aspects x h2

        # Non-Linearity: ReLU
        H_i = F.relu(H_i)                                                           # bsz x num_aspects x h2

        # Item Aspect-level Importance
        # (bsz x num_aspects x h2) * (h2 x 1) -> bsz x num_aspects x 1
        itemAspImpt = torch.matmul(H_i, self.w_hi)                                  # bsz x num_aspects x 1
        if verbose > 0:
            tqdm.write("\nitemAspImpt: {}".format(itemAspImpt.size()))

        itemAspImpt = F.softmax(itemAspImpt, dim=1)                                 # bsz x num_aspects x 1
        if verbose > 0:
            tqdm.write("itemAspImpt: {}".format(itemAspImpt.size()))

        itemAspImpt = torch.squeeze(itemAspImpt, 2)                                 # bsz x num_aspects, beta_i
        if verbose > 0:
            tqdm.write("itemAspImpt: {}".format(itemAspImpt.size()))
        # =========== Item Importance (over Aspects) ===========

        if verbose > 0:
            tqdm.write(
                "\n[Output of AIE] userAspImpt (i.e. the User Aspect-level Importance): {}".format(userAspImpt.size()))
            tqdm.write(
                "[Output of AIE] itemAspImpt (i.e. the Item Aspect-level Importance): {}".format(itemAspImpt.size()))
            tqdm.write(
                "============================== ================================== ==============================\n")

        return userAspImpt, itemAspImpt
