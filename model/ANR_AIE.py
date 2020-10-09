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
        self.W_u = nn.Parameter(torch.Tensor(self.args.h2, self.args.h1), requires_grad=True)   # W_x: h2 x h1
        self.w_hu = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad=True)             # v_x: h2 x 1

        # Item "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_i = nn.Parameter(torch.Tensor(self.args.h2, self.args.h1), requires_grad=True)   # W_y: h2 x h1
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

        userAspRepTrans = torch.transpose(userAspRep, 1, 2)
        itemAspRepTrans = torch.transpose(itemAspRep, 1, 2)
        if verbose > 0:
            tqdm.write("\nuserAspRepTrans: {}".format(userAspRepTrans.size()))      # bsz x h1 x num_aspects
            tqdm.write("itemAspRepTrans: {}".format(itemAspRepTrans.size()))        # bsz x h1 x num_aspects

        '''
        Affinity Matrix (User Aspects x Item Aspects), i.e. User Aspects - Rows, Item Aspects - Columns
        S = RELU(P_u * W_s * Q_i^T)
        '''
        # (bsz x num_aspects x h1) * (h1 x h1) -> bsz x num_aspects x h1
        affinityMatrix = torch.matmul(userAspRep, self.W_a)
        if verbose > 0:
            tqdm.write("\naffinityMatrix: {}".format(affinityMatrix.size()))        # bsz x num_aspects x h1

        # (bsz x num_aspects x h1) * (bsz x h1 x num_aspects) -> bsz x num_aspects x num_aspects
        affinityMatrix = torch.matmul(affinityMatrix, itemAspRepTrans)
        if verbose > 0:
            tqdm.write("affinityMatrix: {}".format(affinityMatrix.size()))          # bsz x num_aspects x num_aspects

        # Non-Linearity: ReLU
        affinityMatrix = F.relu(affinityMatrix)

        '''
        H_u = RELU(P_u * W_x + S^T * (Q_i * W_y))
        beta_u = softmax(H_u * v_x)
        '''
        # =========== User Importance (over Aspects) ===========
        # (h2 x h1) * (bsz x h1 x num_aspects) -> bsz x h2 x num_aspects
        H_u_1 = torch.matmul(self.W_u, userAspRepTrans)
        H_u_2 = torch.matmul(self.W_i, itemAspRepTrans)

        # (bsz x h2 x num_aspects) * (bsz x num_aspects x num_aspects) -> bsz x h2 x num_aspects
        H_u_2 = torch.matmul(H_u_2, affinityMatrix)
        H_u = H_u_1 + H_u_2                                                         # bsz x h2 x num_aspects

        # Non-Linearity: ReLU
        H_u = F.relu(H_u)                                                           # bsz x h2 x num_aspects, H_u

        # User Aspect-level Importance
        # (1 x h2) * (bsz x h2 x num_aspects) -> bsz x 1 x num_aspects
        userAspImpt = torch.matmul(torch.transpose(self.w_hu, 0, 1), H_u)           # bsz x 1 x num_aspects
        if verbose > 0:
            tqdm.write("\nuserAspImpt: {}".format(userAspImpt.size()))

        # User Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
        userAspImpt = torch.transpose(userAspImpt, 1, 2)
        if verbose > 0:
            tqdm.write("userAspImpt: {}".format(userAspImpt.size()))

        userAspImpt = F.softmax(userAspImpt, dim=1)                                 # bsz x 1 x num_aspects, beta_u
        if verbose > 0:
            tqdm.write("userAspImpt: {}".format(userAspImpt.size()))

        # User Aspect-level Importance: (bsz x num_aspects x 1) -> (bsz x num_aspects)
        userAspImpt = torch.squeeze(userAspImpt, 2)
        if verbose > 0:
            tqdm.write("userAspImpt: {}".format(userAspImpt.size()))
        # =========== User Importance (over Aspects) ===========

        '''
        H_i = RELU(Q_i * W_y + S * (P_u * W_x))
        beta_i = softmax(H_i * v_y)
        '''
        # =========== Item Importance (over Aspects) ===========
        # (h2 x h1) * (bsz x h1 x num_aspects) -> bsz x h2 x num_aspects
        H_i_1 = torch.matmul(self.W_i, itemAspRepTrans)
        H_i_2 = torch.matmul(self.W_u, userAspRepTrans)

        # (bsz x h2 x num_aspects) * (bsz x num_aspects x num_aspects) -> bsz x h2 x num_aspects
        H_i_2 = torch.matmul(H_i_2, torch.transpose(affinityMatrix, 1, 2))
        H_i = H_i_1 + H_i_2                                                         # bsz x h2 x num_aspects

        # Non-Linearity: ReLU
        H_i = F.relu(H_i)                                                           # bsz x h2 x num_aspects

        # Item Aspect-level Importance
        # (1 x h2) * (bsz x h2 x num_aspects) -> bsz x 1 x num_aspects
        itemAspImpt = torch.matmul(torch.transpose(self.w_hi, 0, 1), H_i)           # bsz x 1 x num_aspects
        if verbose > 0:
            tqdm.write("\nitemAspImpt: {}".format(itemAspImpt.size()))

        # Item Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
        itemAspImpt = torch.transpose(itemAspImpt, 1, 2)
        if verbose > 0:
            tqdm.write("itemAspImpt: {}".format(itemAspImpt.size()))

        itemAspImpt = F.softmax(itemAspImpt, dim=1)                                 # # bsz x 1 x num_aspects, beta_i
        if verbose > 0:
            tqdm.write("itemAspImpt: {}".format(itemAspImpt.size()))

        # Item Aspect-level Importance: (bsz x num_aspects x 1) -> (bsz x num_aspects)
        itemAspImpt = torch.squeeze(itemAspImpt, 2)
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
