import torch
import torch.nn as nn
import torch.nn.functional as F

from .DenseCNN_RatingPred import DenseCNN_RatingPred

from tqdm import tqdm


class DenseCNN(nn.Module):
    """
    2018. Densely Connected CNN with Multi-scale Feature Attention for Text Classification
    This implementation is base on https://github.com/wangshy31/Densely-Connected-CNN-with-Multiscale-Feature-Attention.git
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

        # # ========== new ==========
        # # Word Embedding Projection Matrices
        # self.wedProj = nn.Parameter(torch.Tensor(self.args.word_embed_dim, self.args.output_size), requires_grad=True)
        # self.wedProj.data.uniform_(-0.01, 0.01)
        # # ========== new ==========

        self.user_net = DenseNet(logger, args)
        self.item_net = DenseNet(logger, args)

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

        # # ========== new ===========
        # # (bsz x max_doc_len x word_embed_dim) x (word_embed_dim x output_size) -> bsz x max_doc_len x output_size
        # batch_userDocEmbed = torch.matmul(batch_userDocEmbed, self.wedProj)
        # batch_itemDocEmbed = torch.matmul(batch_itemDocEmbed, self.wedProj)
        # # ========== new ===========

        batch_userFea = self.user_net(batch_userDocEmbed)
        batch_itemFea = self.item_net(batch_itemDocEmbed)

        # bsz x 1
        rating_pred = self.DenseCNN_RatingPred(batch_userFea, batch_itemFea, batch_uid, batch_iid, verbose=verbose)

        return rating_pred


class DenseNet(nn.Module):
    def __init__(self, logger, args, filters_size=None):
        super(DenseNet, self).__init__()

        self.args = args

        args.filters_num = 1

        if filters_size is None:
            filters_size = [2, 3, 4, 5]

        # bsz x 1 x max_doc_len x output_size -> # bsz x filters_num x max_doc_len x 1
        self.dense_layer_conv0 = nn.Conv2d(1, args.filters_num, kernel_size=(1, args.output_size))
        # bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        self.dense_layer_conv1 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(args.filters_num, args.filters_num, kernel_size=(2, 1)),
            nn.Softmax(dim=2))
        # bsz x filters_num*2 x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        self.dense_layer_conv2 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(args.filters_num*2, args.filters_num, kernel_size=(2, 1)),
            nn.Softmax(dim=2))
        # bsz x filters_num*2 x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        self.dense_layer_conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(args.filters_num*2, args.filters_num, kernel_size=(2, 1)),
            nn.Softmax(dim=2))
        # bsz x filters_num*2 x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        self.dense_layer_conv4 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(args.filters_num*2, args.filters_num, kernel_size=(2, 1)),
            nn.Softmax(dim=2))

        self.activation_layer0 = nn.Sequential(
            nn.BatchNorm2d(args.filters_num),
            nn.ReLU())
        self.activation_layer1 = nn.Sequential(
            nn.BatchNorm2d(args.filters_num * 2),
            nn.ReLU())
        self.activation_layer2 = nn.Sequential(
            nn.BatchNorm2d(args.filters_num * 2),
            nn.ReLU())
        self.activation_layer3 = nn.Sequential(
            nn.BatchNorm2d(args.filters_num * 2),
            nn.ReLU())

        self.scale_attention = nn.Sequential(
            # bsz x filters_num x max_doc_len x 5 -> bsz x 1 x 1 x 5
            nn.Conv2d(args.filters_num, 1, kernel_size=(args.max_doc_len, 1)),
            nn.Softmax(dim=2))

        self.dropout = nn.Dropout(args.dropout_rate)

        self.fcLayer0 = nn.Sequential(
            # bsz x (filters_num x max_doc_len) -> bsz x hidden_size
            nn.Linear(args.filters_num*args.max_doc_len, args.hidden_size),
            nn.Dropout(args.dropout_rate),
            nn.ReLU(),
            # bsz x hidden_size -> bsz x output_size
            nn.Linear(args.hidden_size, args.output_size))

        self.fcLayer1 = nn.Sequential(
            nn.Linear(args.filters_num*args.max_doc_len, args.hidden_size),
            nn.Dropout(args.dropout_rate),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.output_size))

        self.fcLayer2 = nn.Sequential(
            nn.Linear(args.filters_num*args.max_doc_len, args.hidden_size),
            nn.Dropout(args.dropout_rate),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.output_size))

        self.fcLayer3 = nn.Sequential(
            nn.Linear(args.filters_num*args.max_doc_len, args.hidden_size),
            nn.Dropout(args.dropout_rate),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.output_size))

        self.fcLayer4 = nn.Sequential(
            nn.Linear(args.filters_num*args.max_doc_len, args.hidden_size),
            nn.Dropout(args.dropout_rate),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.output_size))

    def reset_para(self):
        convs = [self.dense_layer_conv0, self.dense_layer_conv1[1], self.dense_layer_conv2[1],
                 self.dense_layer_conv3[1], self.dense_layer_conv4[1], self.scale_attention[0]]
        for conv in convs:
            nn.init.xavier_uniform_(conv.weight, gain=1)
            nn.init.uniform_(conv.bias, -0.1, 0.1)

        fcLayers = [self.fcLayer0, self.fcLayer1, self.fcLayer2, self.fcLayer3, self.fcLayer4]
        for fcLayer in fcLayers:
            nn.init.uniform_(fcLayer[0].weight, -0.1, 0.1)
            nn.init.uniform_(fcLayer[-1].weight, -0.1, 0.1)

    '''
    [Input]     x:      bsz x max_doc_len x word_embed_dim
    [Output]    out:    bsz x output_size
    '''
    def forward(self, batch_DocEmbed):
        # bsz x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x word_embed_dim
        batch_DocEmbed = batch_DocEmbed.unsqueeze(1)

        # unigram
        # bsz x 1 x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x 1
        out0 = self.dense_layer_conv0(batch_DocEmbed)
        # bsz x 1 x max_doc_len x 1
        score0 = F.softmax(out0, dim=2)
        # bsz x 1 x max_doc_len x word_embed_dim * bsz x 1 x max_doc_len x 1 -> bsz x 1 x max_doc_len x word_embed_dim
        out0 = torch.mul(batch_DocEmbed, score0)
        # out0 = self.activation_layer0(out0)

        # bigram
        # bsz x 1 x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x word_embed_dim
        score1 = self.dense_layer_conv1(out0)
        # bsz x 1 x max_doc_len x word_embed_dim * bsz x 1 x max_doc_len x word_embed_dim -> bsz x 1 x max_doc_len x word_embed_dim
        out1 = torch.mul(out0, score1)
        # bsz x filters_num*2 x max_doc_len x 1
        out1_0 = torch.cat((out1, out0), dim=1)
        # out1_0 = self.activation_layer1(out1_0)

        # trigram
        # bsz x filters_num*2 x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        score2 = self.dense_layer_conv2(out1_0)
        # bsz x filters_num x max_doc_len x 1 * bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        out2 = torch.mul(out1, score2)
        # bsz x filters_num*2 x max_doc_len x 1
        out2_1 = torch.cat((out2, out1), dim=1)
        # out2_1 = self.activation_layer2(out2_1)

        # fourgram
        # bsz x filters_num*2 x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        score3 = self.dense_layer_conv3(out2_1)
        # bsz x filters_num x max_doc_len x 1 * bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        out3 = torch.mul(out2, score3)
        # bsz x filters_num*2 x max_doc_len x 1
        out3_2 = torch.cat((out3, out2), dim=1)
        # out3_2 = self.activation_layer3(out3_2)

        # fivegram
        # bsz x filters_num*2 x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        score4 = self.dense_layer_conv4(out3_2)
        # bsz x filters_num x max_doc_len x 1 * bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 1
        out4 = torch.mul(out3, score4)

        # bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x 5
        out = torch.cat((out0, out1, out2, out3, out4), dim=3)

        # bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x 1 x 1
        # bsz x filters_num x 1 x 1 -> bsz x filters_num
        out0 = torch.sum(out0, dim=2).squeeze()
        out1 = torch.sum(out1, dim=2).squeeze()
        out2 = torch.sum(out2, dim=2).squeeze()
        out3 = torch.sum(out3, dim=2).squeeze()
        out4 = torch.sum(out4, dim=2).squeeze()

        # bsz x filters_num x 5
        out = torch.stack((out0, out1, out2, out3, out4), dim=2)

        # bsz x filters_num x max_doc_len x 5 -> bsz x 1 x 1 x 5

        # bsz x filters_num x 5 -> bsz x 1 x filters_num x 5
        # bsz x 1 x filters_num x 5 -> bsz x 1 x 1 x 5 -> bsz x 1 x 5
        scale_score = self.scale_attention(out.unsqueeze(1)).squeeze(1)

        # bsz x filters_num x 5 * bsz x 1 x 5 -> bsz x filters_num x 5
        out = torch.mul(out, scale_score)
        # bsz x filters_num x 5 -> bsz x filters_num
        out = torch.sum(out, dim=2)

        return out
