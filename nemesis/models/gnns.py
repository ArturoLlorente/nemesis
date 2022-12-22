import torch
from torch import Tensor
from torch.nn import Linear, Identity, ReLU, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, global_max_pool, BatchNorm, DynamicEdgeConv
from typing import Any, Dict, List, Optional, Union
from torch_geometric.typing import NoneType

class MLP(torch.nn.Module):
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        act: str = "relu",
        batch_norm: bool = True,
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        batch_norm_kwargs: Optional[Dict[str, Any]] = None,
        plain_last: bool = True,
        bias: bool = True,
        relu_first: bool = False,
    ):
        super().__init__()

        act_first = act_first or relu_first  # Backward compatibility.
        batch_norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.dropout = dropout
        self.act = ReLU()
        #self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=bias))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if batch_norm:
                norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
    def forward(self, x: Tensor, return_emb: NoneType = None) -> Tensor:
        """"""
        for lin, norm in zip(self.lins, self.norms):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = x

        if self.plain_last:
            x = self.lins[-1](x)

        return (x, emb) if isinstance(return_emb, bool) else x


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'


class Dynamic_class(torch.nn.Module):
    def __init__(self, out_channels, k, aggr='max'):
        super().__init__()

        #self.conv11 = DynamicEdgeConv(Linear(2*15, 64), k, aggr)
        #self.conv12 = DynamicEdgeConv(Linear(2*64, 64), k, aggr)
        #self.BatchNorm1 = BatchNorm(64)
        #self.pool1 = TopKPooling(64)
#
        #self.conv21 = DynamicEdgeConv(Linear(2*64, 128), k, aggr)
        #self.conv22 = DynamicEdgeConv(Linear(2*128, 128), k, aggr)
        #self.BatchNorm2 = BatchNorm(128)
        #self.pool2 = TopKPooling(128)

        self.conv11 = TAGConv(15, 64, K=4)
        self.conv12 = TAGConv(64, 64, K=4)
        self.BatchNorm1 = BatchNorm(64)

        self.conv21 = TAGConv(64, 64, K=3)
        self.conv22 = TAGConv(64, 32, K=3)
        self.BatchNorm2 = BatchNorm(32)

        self.conv31 = DynamicEdgeConv(Linear(2*32, 64), k, aggr)        
        self.conv32 = DynamicEdgeConv(Linear(2*64, 64), k, aggr)
        self.BatchNorm3 = BatchNorm(64)
        #self.pool3 = TopKPooling(64)

        self.conv41 = DynamicEdgeConv(Linear(2*64, 32), k, aggr)
        self.conv42 = DynamicEdgeConv(Linear(2*32, 32), k, aggr)
        self.BatchNorm4 = BatchNorm(32)
        #self.pool4 = TopKPooling(128)

        #self.feast51 = FeaStConv(128, 128)
        #self.feast52 = FeaStConv(128, 128)
        #self.BatchNorm4 = BatchNorm(128)

        self.lin1 = Linear((64*2 + 32*2), 128)
        self.mlp = MLP([128, 32, out_channels], dropout=0.5)#, norm=None)

    def forward(self, x):#, edge_index, batch):
        x, edge_index, batch = x.x, x.edge_index, x.batch
        x1 = self.conv11(x, edge_index).relu()
        x1 = self.conv12(x1, edge_index).relu()
        x1 = self.BatchNorm1(x1)
        x11 = global_max_pool(x1, batch)        

        x2 = self.conv21(x1, edge_index).relu()
        x2 = self.conv22(x2, edge_index).relu()
        x2 = self.BatchNorm2(x2)
        x21 = global_max_pool(x2, batch)

        x3 = self.conv31(x2, batch)
        x3 = self.conv32(x3, batch)
        x3 = self.BatchNorm3(x3)
        x31 = global_max_pool(x3, batch)

        x4 = self.conv41(x3, batch)
        x4 = self.conv42(x4, batch)
        x4 = self.BatchNorm4(x4)
        x41 = global_max_pool(x4, batch)
    
        out = self.lin1(torch.cat([x11, x21, x31, x41], dim=1))


        out = self.mlp(out)
        return F.log_softmax(out, dim=1)



#class DynEdge_modified(GNN):
#    def __init__(self, input_features, output_features, k = 15, features_subset = slice(12, 15), layer_size_scale=4):
#        
#        #Architecture configuration
#        c = layer_size_scale
#        l1, l2, l3, l4, l5, l6 = (
#            input_features,
#            c * 16 * 2,
#            c * 32 * 2,
#            c * 64 * 2,
#            c * 32 * 2,
#            #c * 16 * 2,
#            output_features,
#        )
#        
#        #Base class constructor
#        super().__init__(l1, l6)
#        
#        #Graph conv opertaions
#        #features_subset = slice(0, 3) 
#        
#        #First Layer
#        self.conv_add1 = DynEdgeConv(
#            torch.nn.Sequential(
#                Linear(l1*2, l2),
#                LeakyReLU(),
#                Linear(l2, l3),
#                LeakyReLU(),
#            ),
#            aggr="max",
#            nb_neighbors=k,
#            features_subset=features_subset,            
#        )              
#          
#        #Second Layer  
#        self.conv_add2 = DynEdgeConv(
#            torch.nn.Sequential(
#                Linear(l3*2, l4),
#                LeakyReLU(),
#                Linear(l4, l3),
#                LeakyReLU(),
#            ),
#            aggr="add",
#            nb_neighbors=k,
#            features_subset=features_subset,            
#        )
#        
#        #Third Layer
#        self.conv_add3 = DynEdgeConv(
#            torch.nn.Sequential(
#                Linear(l3*2, l4),
#                LeakyReLU(),
#                Linear(l4, l3),
#                LeakyReLU(),
#            ),
#            aggr="max",
#            nb_neighbors=k,
#            features_subset=features_subset,            
#        )
#        
#        #Fourth Layer
#        self.conv_cat1 = DynEdgeConv(
#            torch.nn.Sequential(
#                Linear(l3*2, l4),
#                LeakyReLU(),
#                Linear(l4, l3),
#                LeakyReLU(),
#            ),
#            aggr="add",
#            nb_neighbors=k,
#            features_subset=features_subset,            
#        )
#        
#        #Linear layers
#        
#        self.nn1 = Linear(l3*4 + l1, l4)
#        self.nn2 = Linear(l4, l5)
#        self.nn3 = Linear(4*l5 + 3, l6)
#        self.lrelu = LeakyReLU()
#        
#        
#    def forward(self, data):
#        x, edge_index, batch = data.x, data.edge_index, data.batch
#        
#        h_x, h_y, h_z = calculate_xyz_homophily_POne(x, edge_index, batch)
#        
#        global_means = scatter_mean(x, batch, dim=0)
#        
#        a, edge_index = self.conv_add1(x, edge_index, batch)
#        b, edge_index = self.conv_add2(a, edge_index, batch)
#        c, edge_index = self.conv_add3(b, edge_index, batch)
#        d, edge_index = self.conv_cat1(c, edge_index, batch)
#        
#        #Skip cat
#        x = torch.cat((x, a, b, c, d), dim=1)
#        
#        #Post-processing
#        x = self.nn1(x)
#        x = self.lrelu(x)
#        x = self.nn2(x)
#        
#        #Aggregation across nodes
#        a, _ = scatter_max(x, batch, dim=0)
#        b, _ = scatter_min(x, batch, dim=0)
#        c = scatter_sum(x, batch, dim=0)
#        d = scatter_mean(x, batch, dim=0)
#        
#        #Cat aggr and scalar feats
#        
#        x = torch.cat((a, b, c, d, h_x, h_y, h_z), dim=1)
#        
#        #Readout
#        
#        x = self.lrelu(x)
#        x = self.nn3(x)
#        
#        x = self.lrelu(x)
#        
#        return x
    
    
    