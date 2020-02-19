import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
#import torchsnooper 
import numpy as np

from FC import simpleNet, Activation_Net

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2, eps=1e-12)
        return y



class SoftPoolingGcnEncoder(nn.Module):
    def __init__(self, max_num_nodes, num_aspect, multi_conv, multi_pool, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=[0.25,0.25,0.25], assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''
        super(SoftPoolingGcnEncoder, self).__init__()
        print("cat")
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1
        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.aspect = num_aspect
        self.multi_conv = multi_conv
        self.multi_pool = multi_pool
        self.act = nn.ReLU()
        self.label_dim = label_dim

        self.fc_dim = hidden_dim[0] * (num_layers - 1) + embedding_dim[0]
        self.fc_p_dim = int(assign_hidden_dim[0] * (num_layers - 1) + max_num_nodes * assign_ratio[0])
        if self.multi_conv ==1:
            self.FC_first = Activation_Net(3,20,self.aspect)
            self.FC_after_conv = torch.nn.ModuleList()
            for i in range(num_pooling):
                self.FC_after_conv.append(Activation_Net(3,20, self.aspect)) 
        if self.multi_pool == 1:
            self.FC_pool = torch.nn.ModuleList()
            for i in range(num_pooling):
                self.FC_pool.append(Activation_Net(3, 20, self.aspect))

        # multi convolution

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                num_aspect,
                input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
        conv_merge_fist_dim = 0
        for i in range(self.aspect):
            conv_merge_fist_dim = conv_merge_fist_dim + hidden_dim[i] * (num_layers - 1) + embedding_dim[i]
        self.conv_merge_first = self.build_pred_layers(
                conv_merge_fist_dim, [], [int(conv_merge_fist_dim / self.aspect)])
        if multi_conv == 1:
            self.pred_input_dim = []
            for i in range(num_aspect):
                self.pred_input_dim.append(int(conv_merge_fist_dim / self.aspect))
        else:
            self.pred_input_dim = []
            temp_input_dim = hidden_dim[0] * (num_layers - 1) + embedding_dim[0]
            for i in range(num_aspect):
                self.pred_input_dim.append(temp_input_dim)
        if concat:
            self.pred_model = self.build_pred_layers(self.pred_input_dim[0] * (num_pooling + 1), pred_hidden_dims,
                                                     [label_dim], num_aggs=self.num_aggs)
        else:
            self.pred_model = self.build_pred_layers(self.pred_input_dim[0], pred_hidden_dims,
                                                     [label_dim], num_aggs=self.num_aggs)

        self.multi_conv_first_after_pool = torch.nn.ModuleList()
        self.multi_conv_block_after_pool = torch.nn.ModuleList()
        self.multi_conv_last_after_pool = torch.nn.ModuleList()
        for i in range(num_pooling):
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.aspect, self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.multi_conv_first_after_pool.append(conv_first2)
            self.multi_conv_block_after_pool.append(conv_block2)
            self.multi_conv_last_after_pool.append(conv_last2)

        self.embed_merge_conv = nn.ModuleList()
        for i in range(num_pooling):
            embed_conv = self.build_pred_layers(
                conv_merge_fist_dim, [], [int(conv_merge_fist_dim / self.aspect)])
            self.embed_merge_conv.append(embed_conv)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = []
            for i in range(num_aspect):
                assign_input_dim.append(input_dim)

        self.multi_assign_conv_first_modules = torch.nn.ModuleList()
        self.multi_assign_conv_block_modules = torch.nn.ModuleList()
        self.multi_assign_conv_last_modules = torch.nn.ModuleList()
        self.multi_assign_pred_modules = torch.nn.ModuleList()
        self.assign_merge_list = nn.ModuleList()
        assign_dim = []
        for i in range(num_aspect):
            assign_dim.append(int(max_num_nodes * assign_ratio[i]))
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_dim_aspect = []
            for k in range(self.aspect):
                assign_dim_aspect.append(assign_dim[k])
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    num_aspect, assign_input_dim, assign_hidden_dim, assign_dim_aspect, assign_num_layers,
                    add_self, normalize=True)

            assign_pred_input_dim = []
            for j in range(num_aspect):
                temp_assign_pred_input_dim = assign_hidden_dim[j] * (num_layers - 1) + assign_dim[j]  # if concat else assign_dim[j]
                assign_pred_input_dim.append(temp_assign_pred_input_dim)
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1, aspect=self.aspect)
            assign_merge_layer = self.build_pred_layers(
                sum(assign_dim), [], [int((sum(assign_dim)) / self.aspect)])
            self.assign_merge_list.append(assign_merge_layer)

            # next pooling layer
            assign_input_dim = []
            for j in self.pred_input_dim:
                assign_input_dim.append(j)
            for k in range (self.aspect):
                assign_dim[k] = int(assign_dim[k] * assign_ratio[k])
            self.multi_assign_conv_first_modules.append(assign_conv_first)
            self.multi_assign_conv_block_modules.append(assign_conv_block)
            self.multi_assign_conv_last_modules.append(assign_conv_last)
            self.multi_assign_pred_modules.append(assign_pred)

        self.fc_m_normalization= nn.BatchNorm1d(3, affine=True).cuda()
        self.fc_p_normalization = nn.BatchNorm1d(3, affine=True).cuda()

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    #@torchsnooper.snoop()
    def forward(self, x, adj, batch_num_nodes=None, features=None, **kwargs):
        '''
        :param x:
        :param adj:
        :param batch_num_nodes: 
        :param graphfeatures:
        :param kwargs:
        :return:
        '''
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        out_all = []
        if self.multi_conv == 1:
            if len(features)>1:
                fc = self.fc_m_normalization(features)
                fc = self.FC_first(fc)
            else:
                fc = self.FC_first(features)

            fc = nn.Softmax(dim=-1)(fc)
            embedding_tensor_list = self.gcn_forward(x, adj,
                                                self.conv_first, self.conv_block, self.conv_last,
                                                embedding_mask, aspect=self.aspect)

            embedding_tensor = torch.zeros_like(torch.cat(embedding_tensor_list, 2))
            p1 = 0
            for s in range(self.aspect):
                p2 = p1 + embedding_tensor_list[s].size()[2]
                for t in range(len(fc)):
                    embedding_tensor[t, :, p1:p2] = embedding_tensor_list[s][t]*fc[t][s]
                p1 = p2
            embedding_tensor = self.conv_merge_first(embedding_tensor)
        else:
            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first, self.conv_block, self.conv_last,
                                                embedding_mask)
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)
        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None
            if self.multi_pool == 1:
                self.assign_tensor_list = self.gcn_forward(x_a, adj,
                                                  self.multi_assign_conv_first_modules[i],
                                                  self.multi_assign_conv_block_modules[i],
                                                  self.multi_assign_conv_last_modules[i],
                                                  embedding_mask, aspect=self.aspect)

                for k in range(self.aspect):
                    self.assign_tensor_list[k] = nn.Softmax(dim=-1)(
                                                 self.multi_assign_pred_modules[i][k](self.assign_tensor_list[k]))
                if len(features) > 1:
                    fc_p = self.fc_m_normalization(features)
                    fc_p = self.FC_pool[i](fc_p)
                else:
                #fc_p = self.fc_p_normalization(features)
                    fc_p = self.FC_pool[i](features)
                fc_p = nn.Softmax(dim=-1)(fc_p)
                self.assign_tensor = torch.zeros_like(torch.cat(self.assign_tensor_list, 2))

                p1=0
                for s in range(self.aspect):
                    p2 = p1 + self.assign_tensor_list[s].size()[2]
                    for t in range(len(fc_p)):
                        self.assign_tensor[t, :, p1:p2] = self.assign_tensor_list[s][t] * fc_p[t][s]
                self.assign_tensor = nn.Softmax(dim=-1)(
                                     self.assign_merge_list[i](self.assign_tensor))


            else:
                self.assign_tensor = self.gcn_forward(x_a, adj,
                                                      self.multi_assign_conv_first_modules[i],
                                                      self.multi_assign_conv_block_modules[i],
                                                      self.multi_assign_conv_last_modules[i],
                                                      embedding_mask)
                # [batch_size x num_nodes x next_lvl_num_nodes]

                self.assign_tensor = nn.Softmax(dim=-1)(
                    self.multi_assign_pred_modules[i][0](self.assign_tensor))

            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x
            if self.multi_conv == 1:
                if len(features) > 1:
                    fc = self.fc_m_normalization(features)
                    fc = self.FC_after_conv[i](fc)
                else:
                    fc = self.FC_after_conv[i](features)
                #fc = self.fc_m_normalization(features)
                #fc = self.FC_after_conv[i](fc)

                fc = nn.Softmax(dim=-1)(fc)
                embedding_tensor_list = self.gcn_forward(x, adj, self.multi_conv_first_after_pool[i],
                                                         self.multi_conv_block_after_pool[i],
                                                         self.multi_conv_last_after_pool[i], aspect=self.aspect)

                embedding_tensor = torch.zeros_like(torch.cat(embedding_tensor_list, 2))
                p1 = 0
                for s in range(self.aspect):
                    p2 = p1 + embedding_tensor_list[s].size()[2]
                    for t in range(len(fc)):
                        embedding_tensor[t, :, p1:p2] = embedding_tensor_list[s][t] * fc[t][s]
                    p1 = p2

                embedding_tensor = self.embed_merge_conv[i](embedding_tensor)
            else:
                embedding_tensor = self.gcn_forward(x, adj,
                                                    self.multi_conv_first_after_pool[i],
                                                    self.multi_conv_block_after_pool[i],
                                                    self.multi_conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def _loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, size_average=True)
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = self._loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.Tensor(1).cuda())
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[1 - adj_mask.byte()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss



    # @torchsnooper.snoop()
    def build_conv_layers(self, num_aspect, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):

        conv_first = torch.nn.ModuleList()
        conv_block = torch.nn.ModuleList()
        conv_last = torch.nn.ModuleList()
        for i in range(num_aspect):
            in_dim = input_dim[i]
            out_dim = hidden_dim[i]
            conv_first_temp = GraphConv(input_dim=in_dim, output_dim=out_dim, add_self=add_self,
                                        normalize_embedding=normalize, bias=self.bias)
            conv_first.append(conv_first_temp)
            in_dim = hidden_dim[i]
            out_dim = hidden_dim[i]
            conv_block_temp = nn.ModuleList(
                      [GraphConv(input_dim=in_dim, output_dim=out_dim, add_self=add_self,
                                 normalize_embedding=normalize, dropout=dropout, bias=self.bias)
                       for i in range(num_layers-2)])
            conv_block.append(conv_block_temp)

            in_dim = hidden_dim[i]
            out_dim = embedding_dim[i]
            conv_last_temp = GraphConv(input_dim=in_dim, output_dim=out_dim, add_self=add_self,
                                       normalize_embedding=normalize, bias=self.bias)
            conv_last.append(conv_last_temp)

        return conv_first, conv_block, conv_last


    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1, aspect=1):
        if aspect == 1:
            pred_input_dim = pred_input_dim * num_aggs
            if len(pred_hidden_dims) == 0:
                pred_model = nn.Linear(pred_input_dim, label_dim[0])
            else:
                pred_layers = []
                for pred_dim in pred_hidden_dims:
                    pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                    pred_layers.append(self.act)
                    pred_input_dim = pred_dim
                pred_layers.append(nn.Linear(pred_dim, label_dim[0]))
                pred_model = nn.Sequential(*pred_layers)
            return pred_model
        else:
            num_aspect = len(pred_input_dim)
            pred_model = torch.nn.ModuleList()
            for i in range(num_aspect):
                temp_pred_input_dim = pred_input_dim[i] * num_aggs
                if len(pred_hidden_dims) == 0:
                    temp_pred_model = nn.Linear(temp_pred_input_dim, label_dim[i])
                else:
                    pred_layers = []
                    for pred_dim in pred_hidden_dims:
                        pred_layers.append(nn.Linear(temp_pred_input_dim, pred_dim))
                        pred_layers.append(self.act)
                        temp_pred_input_dim = pred_dim
                    pred_layers.append(nn.Linear(pred_dim, label_dim[i]))
                    temp_pred_model = nn.Sequential(*pred_layers)
                pred_model.append(temp_pred_model)
            return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    #@torchsnooper.snoop()
    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last,embedding_mask=None, aspect=1):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        if aspect == 1:
            x = conv_first[0](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all = [x]
            temp_conv_block = conv_block[0]
            for i in range(len(temp_conv_block)):
                x = temp_conv_block[i](x, adj)
                x = self.act(x)
                if self.bn:
                    x = self.apply_bn(x)
                x_all.append(x)
            x = conv_last[0](x, adj)
            x_all.append(x)
            # x_tensor: [batch_size x num_nodes x embedding]
            x_tensor = torch.cat(x_all, dim=2)
            if embedding_mask is not None:
                x_tensor = x_tensor * embedding_mask
        else:
            xlist = []
            adjlist = []
            length = len(conv_first)
            for i in range(length):
                xlist.append((x))
                adjlist.append(adj)
            x_tensor = []
            for i in range(length):
                x = xlist[i]
                adj = adjlist[i]
                x = conv_first[i](x, adj)
                x = self.act(x)
                if self.bn:
                    x = self.apply_bn(x)
                x_all = [x]
                temp_conv_block = conv_block[i]
                for j in range(len(temp_conv_block)):
                    x = temp_conv_block[j](x, adj)
                    x = self.act(x)
                    if self.bn:
                        x = self.apply_bn(x)
                    x_all.append(x)
                x = conv_last[i](x, adj)
                x_all.append(x)
                temp_x_tensor = torch.cat(x_all, dim=2)
                if embedding_mask is not None:
                    temp_x_tensor = temp_x_tensor * embedding_mask
                x_tensor.append(temp_x_tensor)

        return x_tensor

