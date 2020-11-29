import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import to_var

from tqdm import tqdm


class MSANR_ARL(nn.Module):
    """
    Multi Scale Aspect-based Representation Learning (MSARL)
    """

    def __init__(self, logger, args):
        super(MSANR_ARL, self).__init__()

        self.logger = logger
        self.args = args

        # num_aspects x (ctx_win_size x h1)
        self.aspEmbed = nn.ModuleList([nn.Embedding(self.args.num_aspects, c * self.args.h1) for c in self.args.kernel_list])
        for aspEmbed in self.aspEmbed:
            aspEmbed.weight.requires_grad = True

        # num_aspects x h1
        self.filterEmbed = nn.Embedding(self.args.num_aspects, self.args.h1)

        # num_aspects x word_embed_dim x h1
        self.aspProj = nn.Parameter(torch.Tensor(self.args.num_aspects, self.args.word_embed_dim, self.args.h1),
                                    requires_grad=True)

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        for aspEmbed in self.aspEmbed:
            aspEmbed.weight.data.uniform_(-0.01, 0.01)
        self.filterEmbed.weight.data.uniform_(-0.01, 0.01)
        self.aspProj.data.uniform_(-0.01, 0.01)

    '''
    [Input]     batch_docIn:    bsz x max_doc_len x word_embed_dim
    [Output]    batch_aspRep:   bsz x num_aspects x h1
    '''
    def forward(self, batch_docIn, verbose=0):
        # Loop over all aspects
        lst_batch_aspAttn = []
        lst_batch_aspRep = []
        for a in range(self.args.num_aspects):
            lst_batch_ctxAttn = []
            lst_batch_ctxRep = []
            for i, c in enumerate(self.args.kernel_list):
                # Aspect-Specific Projection of Input Word Embeddings
                # (bsz x max_doc_len x word_embed_dim) * (word_embed_dim x h1) -> bsz x max_doc_len x h1
                batch_aspProjDoc = torch.matmul(batch_docIn, self.aspProj[a])                           # bsz x max_doc_len x h1

                # Aspect Embedding: (bsz x (ctx_win_size x h1) x 1) after tranposing!
                bsz = batch_docIn.size()[0]
                # bsz x 1 x (ctx_win_size x h1)
                batch_aspEmbed = self.aspEmbed[i](to_var(torch.LongTensor(bsz, 1).fill_(a), use_cuda=self.args.use_cuda))
                # bsz x (ctx_win_size x h1) x 1
                batch_aspEmbed = torch.transpose(batch_aspEmbed, 1, 2)

                # Window Size (self.args.ctx_win_size) of 1: Calculate Attention based on the word itself!
                if c == 1:
                    # Calculate Attention: Inner Product & Softmax
                    # (bsz x max_doc_len x h1) x (bsz x h1 x 1) -> (bsz x max_doc_len x 1)
                    batch_aspAttn = torch.matmul(batch_aspProjDoc, batch_aspEmbed)
                    batch_aspAttn = F.softmax(batch_aspAttn, dim=1)

                # Context-based Word Importance
                # Calculate Attention based on the word itself, and the (self.args.ctx_win_size - 1) / 2 word(s) before & after it
                else:
                    # Pad the document
                    pad_size = int((c - 1) / 2)
                    if c % 2 == 0:
                        batch_aspProjDoc_padded = F.pad(batch_aspProjDoc, [0, 0, pad_size, pad_size+1], "constant", 0)
                    else:
                        # bsz x (max_doc_len+2) x h1
                        batch_aspProjDoc_padded = F.pad(batch_aspProjDoc, [0, 0, pad_size, pad_size], "constant", 0)

                    # Use "sliding window" using stride of 1 (word at a time) to generate word chunks of ctx_win_size
                    # bsz x (max_doc_len+2) x h1 -> bsz x max_doc_len x h1 x ctx_win_size
                    batch_aspProjDoc_padded = batch_aspProjDoc_padded.unfold(1, c, 1)
                    # bsz x max_doc_len x ctx_win_size x h1
                    batch_aspProjDoc_padded = torch.transpose(batch_aspProjDoc_padded, 2, 3)
                    # bsz x max_doc_len x (ctx_win_size x h1)
                    batch_aspProjDoc_padded = batch_aspProjDoc_padded.contiguous().view(-1, self.args.max_doc_len,
                                                                                        c * self.args.h1)

                    # Calculate Attention: Inner Product & Softmax
                    # bsz x max_doc_len x (ctx_win_size x h1) * bsz x (ctx_win_size x h1) x 1 -> bsz x max_doc_len x 1
                    batch_aspAttn = torch.matmul(batch_aspProjDoc_padded, batch_aspEmbed)
                    batch_aspAttn = F.softmax(batch_aspAttn, dim=1)

                # Weighted Sum: Broadcasted Element-wise Multiplication & Sum over Words
                # bsz x max_doc_len x h1 * bsz x max_doc_len x 1 -> bsz x max_doc_len x h1
                batch_aspRep = torch.mul(batch_aspProjDoc, batch_aspAttn)
                # bsz x max_doc_len x h1 -> bsz x h1
                batch_aspRep = torch.sum(batch_aspRep, dim=1)

                # [bsz x 1 x max_doc_len]
                lst_batch_ctxAttn.append(torch.transpose(batch_aspAttn, 1, 2))
                # [bsz x h1]
                lst_batch_ctxRep.append(batch_aspRep)
            # bsz x filters x max_doc_len
            batch_ctxAttn = torch.cat(lst_batch_ctxAttn, dim=1)
            # bsz x filters x h1
            batch_ctxRep = torch.stack(lst_batch_ctxRep, dim=1)

            bsz = batch_docIn.size()[0]
            # bsz x 1 x h1
            batch_filterEmbed = self.filterEmbed(to_var(torch.LongTensor(bsz, 1).fill_(a), use_cuda=self.args.use_cuda))
            # bsz x h1 x 1
            batch_filterEmbed = torch.transpose(batch_filterEmbed, 1, 2)
            # (bsz x filters x h1) x (bsz x h1 x 1) -> (bsz x filters x 1)
            batch_filterAttn = torch.matmul(batch_ctxRep, batch_filterEmbed)

            # bsz x filters x h1 * bsz x filters x 1 -> bsz x filters x h1
            batch_aspRep = torch.mul(batch_ctxRep, batch_filterAttn)
            # bsz x filters x h1 -> bsz x h1
            batch_aspRep = torch.sum(batch_aspRep, dim=1)

            # bsz x filters x max_doc_len * bsz x filters x 1 -> bsz x filters x max_doc_len
            batch_aspAttn = torch.mul(batch_ctxAttn, batch_filterAttn)
            # bsz x filters x max_doc_len -> bsz x max_doc_len
            batch_aspAttn = torch.sum(batch_aspAttn, dim=1)

            # Store the results (Attention & Representation) for this aspect
            lst_batch_aspRep.append(batch_aspRep)
            lst_batch_aspAttn.append(batch_aspAttn)

        # Reshape the Attentions & Representations
        # batch_aspAttn:        bsz x num_aspects x max_doc_len
        # batch_aspRep:         bsz x num_aspects x h1
        batch_aspRep = torch.stack(lst_batch_aspRep, dim=1)
        batch_aspAttn = torch.stack(lst_batch_aspAttn, dim=1)

        # Returns the aspect-level attention over document words, and the aspect-based representations
        return batch_aspAttn, batch_aspRep
