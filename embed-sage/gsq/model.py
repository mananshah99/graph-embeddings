import torch
import torch.nn as nn
import math
from torch.nn import init
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from gsq.toy_graph import get_toy_graph

EPS = 10e-6

"""
Methods for representation learning on
heteregenous graphs/networks
"""

class PathEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons over metapaths
    """

    def __init__(self, graph, enc, dec):
        super(PathEncoderDecoder, self).__init__()
        self.enc = enc
        self.dec = dec
        self.graph = graph

    def forward(self, nodes1, nodes2, rels):
        return self.dec.forward(self.enc.forward(nodes1, rels[0][0]).squeeze(), 
                self.enc.forward(nodes2, rels[-1][-1]).squeeze(),
                rels)

    def margin_loss(self, nodes1, nodes2, rels):
        neg_nodes = self.graph.adj_lists[rels[0]].keys()
        affs = self.forward(nodes1, nodes2, rels)
        neg_nodes = np.random.choice(neg_nodes, size=len(nodes1))
        neg_affs = self.forward(neg_nodes.tolist(), nodes2,
            rels)
        margin = 1 - (affs - neg_affs)
        margin = torch.clamp(margin, min=0)
        loss = margin.mean()
        return loss 


class EdgeEncoderDecoder(nn.Module):
    """
    Encoder-decoder model that reasons over edges.
    (i.e., link prediction model)
    """

    def __init__(self, graph, enc, dec):
        super(EdgeEncoderDecoder, self).__init__()
        self.enc = enc
        self.dec = dec
        self.graph = graph
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, nodes1, nodes2, rel):
        return self.dec.forward(self.enc.forward(nodes1, rel[0]),
                self.enc.forward(nodes2, rel[2]),
                rel)

    def margin_loss(self, nodes1, nodes2, rel):
        neg_nodes = self.graph.adj_lists[rel].keys()
        affs = self.forward(nodes1, nodes2, rel)
        neg_nodes = np.random.choice(neg_nodes, size=len(nodes1))
        neg_affs = self.forward(neg_nodes.tolist(), nodes2,
            rel)
        margin = 1 - (affs - neg_affs)
        margin = torch.clamp(margin, min=0)
        loss = margin.mean()
        return loss 


class DirectEncoder(nn.Module):
    """
    Encodes a node as a embedding via direct lookup.
    (i.e., this is just like basic node2vec or matrix factorization)
    """
    def __init__(self, features, feature_modules): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_modules  -- This should be a map from mode -> torch.nn.EmbeddingBag 
        """
        super(DirectEncoder, self).__init__()
        for name, module in feature_modules.iteritems():
            self.add_module("feat-"+name, module)
        self.features = features

    def single_forward(self, node, mode):
        return self.features(node, mode)

    def forward(self, nodes, mode, offset=None):
        """
        Generates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        offsets   -- specifies how the embeddings are aggregated. 
                     see torch.nn.EmbeddingBag for format. 
                     No aggregation if offsets is None
        """

        if offset is None:
            return self.features(nodes, mode, range(len(nodes))).t()
        else:
            return self.features(nodes, mode, offset).t()

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' approach
    """
    def __init__(self, features, feature_dims, 
            out_dims, relations, adj_lists, 
            base_model=None, concat=False, cuda=False,
            feature_modules={}): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_dims     -- output dimension of each of the feature functions. 
        out_dims       -- embedding dimensions for each mode (i.e., output dimensions)
        relations        -- map from mode -> out_going_relations
        adj_lists        -- map from relation_tuple -> node -> list of node's neighbors
        base_model       -- if features are from another encoder, pass it here for training
        concat           -- whether to concat or sum information from different modes
        cuda             -- whether or not to move params to the GPU
        feature_modules  -- if features come from torch.nn module, pass the modules here for training
        """

        super(Encoder, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.adj_lists = adj_lists
        self.relations = relations
        for name, module in feature_modules.iteritems():
            self.add_module("feat-"+name, module)
        if base_model != None:
            self.base_model = base_model

        self.concat = concat
        self.out_dims = out_dims
        self.cuda = cuda
        if not concat:
            if len(set(out_dims.values())) != 1:
                raise Exception("Inner dimensions must all be equal if concat=False")
        else:
            self.weight_dims = {}
            for source_mode in relations:
                self.weight_dims[source_mode] = out_dims[source_mode]
                for (to_mode, _) in relations[source_mode]:
                    self.weight_dims[source_mode] += out_dims[to_mode]

        self.self_params = {}
        self.compress_params = {}
        for mode, feat_dim in self.feat_dims.iteritems():
            self.self_params[mode] = nn.Parameter(
                    torch.FloatTensor(out_dims[mode], feat_dim))
            init.xavier_uniform(self.self_params[mode])
            self.register_parameter(mode+"_self", self.self_params[mode])
            
            if concat:
                self.compress_params[mode] = nn.Parameter(
                        torch.FloatTensor(out_dims[mode], self.weight_dims[mode]))
                init.xavier_uniform(self.compress_params[mode])
                self.register_parameter(mode+"_compress", self.compress_params[mode])


        self.relation_params = {}
        for from_r, to_rs in relations.iteritems():
            for to_r in to_rs:
                rel = (from_r, to_r[1], to_r[0])
                self.relation_params[rel] = nn.Parameter(
                        torch.FloatTensor(out_dims[from_r], self.feat_dims[to_r[0]]))
                init.xavier_uniform(self.relation_params[rel])
                self.register_parameter("_".join(rel), 
                        self.relation_params[rel])


    def forward(self, nodes, mode, offsets=None, 
            keep_prob=0.75, max_keep=25):
        """
        Generates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        offsets   -- specifies how the embeddings are aggregated. 
                     see torch.nn.EmbeddingBag for format. 
                     No aggregation if offsets is None
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        self_feat = self.features(nodes, mode, range(len(nodes))) 
        self_feat = self.self_params[mode].mm(
                self_feat.t())
        neigh_feats = []
        for to_r in self.relations[mode]:
            rel = (mode, to_r[1], to_r[0])
            to_neighs = [self.adj_lists[rel][node] for node in nodes]

            # Special null neighbor for nodes with no edges of this type
            to_neighs = [[-1] if len(l) == 0 else l for l in to_neighs]
            samp_neighs = [np.random.choice(to_neigh, 
                        min(int(math.ceil(len(to_neigh)*keep_prob)), max_keep),
                        replace=False) for to_neigh in to_neighs]
            to_feats = self.features([node for samp_neigh in samp_neighs for node in samp_neigh],
                    rel[-1], 
                    [0]+np.cumsum([len(samp_neigh) for samp_neigh in samp_neighs])[:-1].tolist())
            to_feats = self.relation_params[rel].mm(to_feats.t())

            neigh_feats.append(to_feats)
        
        neigh_feats.append(self_feat)
        if self.concat:
            combined = torch.cat(neigh_feats, dim=0)
            combined = self.compress_params[mode].mm(combined)
        else:
            combined = torch.stack(neigh_feats, dim=0)
            combined = combined.mean(dim=0).squeeze()
        combined = F.relu(combined)
        if offsets != None:
            if self.cuda:
                return self._backend.EmbeddingBag(
                        None, 2, False, mode='mean')(combined.t().contiguous(), 
                                Variable(torch.LongTensor(np.arange(len(nodes)))).cuda().contiguous(), 
                                Variable(torch.LongTensor(offsets)).cuda().contiguous()).t()
            else:
                return self._backend.EmbeddingBag(
                        None, 2, False, mode='mean')(combined.t(), 
                                Variable(torch.LongTensor(np.arange(len(nodes)))), 
                                Variable(torch.LongTensor(offsets))).t()

        return combined



class BilinearDecoder(nn.Module):
    """
    Decodes edges using a bilinear form.
    Each edge type has its own matrix.
    """

    def __init__(self, relations, dims):
        """
        relations -- map from mode -> relations (outgoing from that mode)
        dims      -- map from mode -> dimension (size of embedddings)
        """
        super(BilinearDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(
                        torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                init.xavier_uniform(self.mats[rel])
                self.register_parameter("_".join(rel), self.mats[rel])

    def forward(self, embeds1, embeds2, rel):
        acts = self.mats[rel].mm(embeds2)
        acts = embeds1 * acts
        return acts.sum(dim=0)

class BilinearPathDecoder(nn.Module):
    """
    Decodes metapaths btwn nodes using bilinear forms to represent relations.
    Each edge type has its own matrix.
    """

    def __init__(self, relations, dims):
        """
        relations -- map from mode -> relations (outgoing from that mode)
        dims      -- map from mode -> dimension (size of embedddings)
        """
        super(BilinearPathDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        self.sigmoid = torch.nn.Sigmoid()
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                init.xavier_uniform(self.mats[rel])
                self.register_parameter("_".join(rel), self.mats[rel])

    def forward(self, embeds1, embeds2, rels):
        """
        Returns the score of this metapath btwn nodes
        embeds1 --- batch of embeddings from a common mode
        embeds2 --- batch of embeddings from a common mode
        rels    --- list of relations (metapath)
                    must start at mode type of embeds1 and end on type of embeds2
        """
        act = embeds1.t()
        for i_rel in rels:
            act = act.mm(self.mats[i_rel])
        act = act.t() * embeds2
        return act.sum(dim=0)

if __name__ == "__main__":
    graph, feature_modules = get_toy_graph()
    out_dims = {"m1" : 3, "m2" : 3}
    enc1 = Encoder(graph.features, 
            graph.feature_dims, 
            out_dims, 
            graph.relations, 
            graph.adj_lists, concat=True, feature_modules=feature_modules)
    #print enc1.batched_forward([0,1], "m2")
    enc2 = Encoder(lambda nodes, mode, offset : enc1.forward(nodes, mode, offset).t(),
            enc1.out_dims, 
            {"m1":3, "m2":3},
            graph.relations, 
            graph.adj_lists,
            base_model=enc1, concat=True)
    dec = BilinearPathDecoder(graph.relations, enc2.out_dims)
    enc_dec = PathEncoderDecoder(graph, enc2, dec)
    print(enc_dec.forward(0, 0, [("m1", "0", "m2"), ("m2", "0", "m2"), ("m2", "0", "m2")]))
