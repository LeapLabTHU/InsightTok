from torch import nn
import torch.nn.functional as F
import torch
import torch.distributed as dist
import numpy as np
from collections import defaultdict


def is_available():
    return dist.is_available() and dist.is_initialized()

def all_reduce(tensor, op=dist.ReduceOp.SUM):
    if is_available():
        dist.all_reduce(tensor, op=op)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, l2_norm, show_usage, restart_mode='random', verbose=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.decay = 0.99
        self.eps = 1e-5
        self.cluster_size = torch.nn.Parameter(torch.ones(n_e), requires_grad = False)
        self.embed_avg = torch.nn.Parameter(self.embedding.weight.clone(), requires_grad = False)
        self.update = True
        self.threshold = 1.0
        self.restart_mode = restart_mode
        if verbose:
            print(f"EMA-VQ restart mode: {self.restart_mode}")
        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(65536))
        
        self.cache = defaultdict(list)


    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_e:
            n_repeats = (self.n_e + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x
    

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))



        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)     
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))
            min_encodings = None

            with torch.no_grad():
                #EMA cluster size
                encodings_sum = encodings.sum(0)
                embed_sum = encodings.transpose(0,1) @ z_flattened  

                all_reduce(encodings_sum)
                all_reduce(embed_sum)
                self.cache['encodings_sum'].append(encodings_sum)
                self.cache['embed_sum'].append(embed_sum)

                if self.restart_mode is not None:
                    if is_available():
                        z_flattened_gathered = [torch.zeros_like(z_flattened.detach()) for _ in range(torch.distributed.get_world_size())]
                        torch.distributed.all_gather(z_flattened_gathered, z_flattened)
                        z_flattened_gathered = torch.cat(z_flattened_gathered, dim=0)
                    else:
                        z_flattened_gathered = z_flattened
                    self.cache['z_flattened'].append(z_flattened_gathered)
                
                if self.restart_mode == 'dist':
                    if is_available():
                        d_gathered = [torch.zeros_like(d.detach()) for _ in range(torch.distributed.get_world_size())]
                        torch.distributed.all_gather(d_gathered, d)
                        d_gathered = torch.cat(d_gathered, dim=0)
                    else:
                        d_gathered = d
                    self.cache['d'].append(d_gathered)
                

            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            vq_loss = commit_loss * 0.0

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)


        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)


    @torch.no_grad()
    def update_embedding(self):
        encodings_sum = sum(self.cache['encodings_sum'], 0)
        embed_sum = sum(self.cache['embed_sum'], 0)
        
        self.cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
        self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        usage = (self.cluster_size.unsqueeze(1) >= self.threshold).float()
        unused_mask = (1-usage.squeeze()).bool()
        
        #normalize embed_avg and update weight
        embed_normalized = self.embed_avg / self.cluster_size.unsqueeze(1)

        if self.restart_mode == None:
            embed_final = embed_normalized
        elif self.restart_mode == "random":
            z_flattened = torch.cat(self.cache['z_flattened'], dim=0)
            # random choose points
            y = self._tile(z_flattened)
            rand_z = y[torch.randperm(y.shape[0])][:self.n_e]
            embed_final = usage * embed_normalized + (1 - usage) * rand_z
        elif self.restart_mode == 'dist':
            z_flattened = torch.cat(self.cache['z_flattened'], dim=0)
            d = torch.cat(self.cache['d'], dim=0)
            # choose points that are farthest from the current embeddings
            min_dist = torch.min(d, dim=-1)[0]
            dist_indices = torch.argsort(min_dist, descending=True)
            num_unused = (1-usage).sum().long()
            dist_indices = dist_indices[:num_unused]
            z_flattened_chosen = self._tile(z_flattened[dist_indices])[:num_unused]
            embed_normalized[unused_mask] = z_flattened_chosen
            embed_final = embed_normalized
        else:
            raise ValueError("Invalid restart mode: {}".format(self.restart_mode))
        
        if self.l2_norm:
            embed_final = F.normalize(embed_final, p=2, dim=-1)
        self.embedding.weight.data.copy_(embed_final)
        self.cache = defaultdict(list)


    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q

