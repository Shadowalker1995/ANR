import torch
import torch.nn as nn
import torch.nn.functional as F

from .DenseCNN_RatingPred import DenseCNN_RatingPred

from tqdm import tqdm


class DenseCNN(nn.Module):
    """
    2018. Densely Connected CNN with Multi-scale Feature Attention for Text Classification
    with token attention mechanism in each densely convolution layer and
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
        self.wedProj = nn.Parameter(torch.Tensor(self.args.word_embed_dim, self.args.output_size), requires_grad=True)
        self.wedProj.data.uniform_(-0.01, 0.01)

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
        batch_userDocEmbed = torch.matmul(batch_userDocEmbed, self.wedProj)
        batch_itemDocEmbed = torch.matmul(batch_itemDocEmbed, self.wedProj)

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
            filters_size = [2, 3, 4, 5]

        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        self.attention_layer_conv0 = nn.Sequential(
            nn.Conv2d(args.filters_num, args.filters_num, kernel_size=(1, args.output_size), groups=args.filters_num),
            nn.Softmax(dim=2))
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        self.attention_layer_conv1 = nn.Sequential(
            nn.Conv2d(args.filters_num, args.filters_num, kernel_size=(1, args.output_size), groups=args.filters_num),
            nn.Softmax(dim=2))
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        self.attention_layer_conv2 = nn.Sequential(
            nn.Conv2d(args.filters_num, args.filters_num, kernel_size=(1, args.output_size), groups=args.filters_num),
            nn.Softmax(dim=2))
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        self.attention_layer_conv3 = nn.Sequential(
            nn.Conv2d(args.filters_num, args.filters_num, kernel_size=(1, args.output_size), groups=args.filters_num),
            nn.Softmax(dim=2))
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        self.attention_layer_conv4 = nn.Sequential(
            nn.Conv2d(args.filters_num, args.filters_num, kernel_size=(1, args.output_size), groups=args.filters_num),
            nn.Softmax(dim=2))

        # bsz x 1 x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        self.dense_layer_conv0 = nn.Sequential(
            nn.Conv2d(1, args.filters_num, kernel_size=(1, 1)),
            nn.BatchNorm2d(args.filters_num),
            nn.ReLU())
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        self.dense_layer_conv1 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(args.filters_num, args.filters_num, kernel_size=(2, 1)),
            nn.BatchNorm2d(args.filters_num),
            nn.ReLU())
        # bsz x 2*filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        self.dense_layer_conv2 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(2*args.filters_num, args.filters_num, kernel_size=(2, 1)),
            nn.BatchNorm2d(args.filters_num),
            nn.ReLU())
        # bsz x 2*filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        self.dense_layer_conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(2*args.filters_num, args.filters_num, kernel_size=(2, 1)),
            nn.BatchNorm2d(args.filters_num),
            nn.ReLU())
        # bsz x 2*filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        self.dense_layer_conv4 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(2*args.filters_num, args.filters_num, kernel_size=(2, 1)),
            nn.BatchNorm2d(args.filters_num),
            nn.ReLU())

        # bsz x filters_num x output_size x 5 -> bsz x filters_num x 1 x 5
        self.scale_attention = nn.Sequential(
            nn.Conv2d(args.filters_num, args.filters_num, kernel_size=(args.output_size, 1), groups=args.filters_num),
            nn.Softmax(dim=3))

        # # bsz x 1 x filters_num x output_size -> bsz x 1 x filters_num x 1
        self.filter_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, args.output_size)),
            nn.Softmax(dim=2))

        self.dropout = nn.Dropout(args.dropout_rate)

    def reset_para(self):
        convs = [self.attention_layer_conv0[0], self.attention_layer_conv1[0], self.attention_layer_conv2[0],
                 self.attention_layer_conv3[0], self.attention_layer_conv4[0],
                 self.dense_layer_conv0[0], self.dense_layer_conv1[1], self.dense_layer_conv2[1],
                 self.dense_layer_conv3[1], self.dense_layer_conv4[1],
                 self.scale_attention[0], self.filter_attention[0]]
        for conv in convs:
            nn.init.xavier_uniform_(conv.weight, gain=1)
            nn.init.uniform_(conv.bias, -0.1, 0.1)

    '''
    [Input]     x:      bsz x max_doc_len x output_size
    [Output]    out:    bsz x output_size
    '''
    def forward(self, batch_DocEmbed):
        # bsz x max_doc_len x output_size -> bsz x 1 x max_doc_len x output_size
        batch_DocEmbed = batch_DocEmbed.unsqueeze(1)

        # unigram
        # bsz x 1 x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        batch_DocEmbed = self.dense_layer_conv0(batch_DocEmbed)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        score0 = self.attention_layer_conv0(batch_DocEmbed)
        # bsz x filters_num x max_doc_len x output_size * bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x output_size
        out0 = torch.mul(batch_DocEmbed, score0)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x output_size
        fea0 = torch.sum(out0, dim=2)
        out0 = self.dropout(out0)

        # bigram
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        out1 = self.dense_layer_conv1(out0)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        score1 = self.attention_layer_conv1(out1)
        # bsz x filters_num x max_doc_len x output_size * bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x output_size
        out1 = torch.mul(out1, score1)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x output_size
        fea1 = torch.sum(out1, dim=2)
        out1 = self.dropout(out1)

        # trigram
        # bsz x 2*filters_num x max_doc_len x output_size
        out1_0 = torch.cat((out1, out0), dim=1)
        # bsz x 2*filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        out1_0 = self.dense_layer_conv2(out1_0)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        score2 = self.attention_layer_conv2(out1_0)
        # bsz x filters_num x max_doc_len x output_size * bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x output_size
        out2 = torch.mul(out1_0, score2)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x output_size
        fea2 = torch.sum(out2, dim=2)
        out2 = self.dropout(out2)

        # fourgram
        # bsz x 2*filters_num x max_doc_len x output_size
        out2_1 = torch.cat((out2, out1), dim=1)
        # bsz x 2*filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        out2_1 = self.dense_layer_conv3(out2_1)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        score3 = self.attention_layer_conv3(out2_1)
        # bsz x filters_num x max_doc_len x output_size * bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x output_size
        out3 = torch.mul(out2_1, score3)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x output_size
        fea3 = torch.sum(out3, dim=2)
        out3 = self.dropout(out3)

        # fivegram
        # bsz x 2*filters_num x max_doc_len x output_size
        out3_2 = torch.cat((out3, out2), dim=1)
        # bsz x 2*filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x output_size
        out3_2 = self.dense_layer_conv4(out3_2)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x max_doc_len x 1
        score4 = self.attention_layer_conv4(out3_2)
        # bsz x filters_num x max_doc_len x output_size * bsz x filters_num x max_doc_len x 1 -> bsz x filters_num x max_doc_len x output_size
        out4 = torch.mul(out3_2, score4)
        # bsz x filters_num x max_doc_len x output_size -> bsz x filters_num x output_size
        fea4 = torch.sum(out4, dim=2)

        # bsz x filters_num x output_size -> bsz x filters_num x output_size x 5
        fea = torch.stack((fea0, fea1, fea2, fea3, fea4), dim=3)

        # bsz x filters_num x output_size x 5 -> bsz x filters_num x 1 x 5
        scale_score = self.scale_attention(fea)

        # bsz x filters_num x output_size x 5 * bsz x filters_num x 1 x 5 -> bsz x filters_num x output_size x 5
        fea = torch.mul(fea, scale_score)
        # bsz x filters_num x output_size x 5 -> bsz x filters_num x output_size
        fea = torch.sum(fea, dim=3)

        # bsz x filters_num x output_size -> bsz x 1 x filters_num x output_size
        # bsz x 1 x filters_num x output_size -> bsz x 1 x filters_num x 1 -> bsz x filters_num x 1
        filter_score = self.filter_attention(fea.unsqueeze(1)).squeeze(1)

        # bsz x filters_num x output_size * bsz x filters_num x 1 -> bsz x filters_num x output_size
        fea = torch.mul(fea, filter_score)
        # bsz x filters_num x output_size -> bsz x output_size
        fea = torch.sum(fea, dim=1)

        return fea
