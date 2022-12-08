

class DynEdge_modified(GNN):
    def __init__(self, input_features, output_features, k = 15, features_subset = slice(12, 15), layer_size_scale=4):
        
        #Architecture configuration
        c = layer_size_scale
        l1, l2, l3, l4, l5, l6 = (
            input_features,
            c * 16 * 2,
            c * 32 * 2,
            c * 64 * 2,
            c * 32 * 2,
            #c * 16 * 2,
            output_features,
        )
        
        #Base class constructor
        super().__init__(l1, l6)
        
        #Graph conv opertaions
        #features_subset = slice(0, 3) 
        
        #First Layer
        self.conv_add1 = DynEdgeConv(
            torch.nn.Sequential(
                Linear(l1*2, l2),
                LeakyReLU(),
                Linear(l2, l3),
                LeakyReLU(),
            ),
            aggr="max",
            nb_neighbors=k,
            features_subset=features_subset,            
        )              
          
        #Second Layer  
        self.conv_add2 = DynEdgeConv(
            torch.nn.Sequential(
                Linear(l3*2, l4),
                LeakyReLU(),
                Linear(l4, l3),
                LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=k,
            features_subset=features_subset,            
        )
        
        #Third Layer
        self.conv_add3 = DynEdgeConv(
            torch.nn.Sequential(
                Linear(l3*2, l4),
                LeakyReLU(),
                Linear(l4, l3),
                LeakyReLU(),
            ),
            aggr="max",
            nb_neighbors=k,
            features_subset=features_subset,            
        )
        
        #Fourth Layer
        self.conv_cat1 = DynEdgeConv(
            torch.nn.Sequential(
                Linear(l3*2, l4),
                LeakyReLU(),
                Linear(l4, l3),
                LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=k,
            features_subset=features_subset,            
        )
        
        #Linear layers
        
        self.nn1 = Linear(l3*4 + l1, l4)
        self.nn2 = Linear(l4, l5)
        self.nn3 = Linear(4*l5 + 3, l6)
        self.lrelu = LeakyReLU()
        
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        h_x, h_y, h_z = calculate_xyz_homophily_POne(x, edge_index, batch)
        
        global_means = scatter_mean(x, batch, dim=0)
        
        a, edge_index = self.conv_add1(x, edge_index, batch)
        b, edge_index = self.conv_add2(a, edge_index, batch)
        c, edge_index = self.conv_add3(b, edge_index, batch)
        d, edge_index = self.conv_cat1(c, edge_index, batch)
        
        #Skip cat
        x = torch.cat((x, a, b, c, d), dim=1)
        
        #Post-processing
        x = self.nn1(x)
        x = self.lrelu(x)
        x = self.nn2(x)
        
        #Aggregation across nodes
        a, _ = scatter_max(x, batch, dim=0)
        b, _ = scatter_min(x, batch, dim=0)
        c = scatter_sum(x, batch, dim=0)
        d = scatter_mean(x, batch, dim=0)
        
        #Cat aggr and scalar feats
        
        x = torch.cat((a, b, c, d, h_x, h_y, h_z), dim=1)
        
        #Readout
        
        x = self.lrelu(x)
        x = self.nn3(x)
        
        x = self.lrelu(x)
        
        return x