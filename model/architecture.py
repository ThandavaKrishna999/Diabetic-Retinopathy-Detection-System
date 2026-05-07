import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    from torchvision.models import MobileNet_V2_Weights, DenseNet121_Weights
    _mobilenet_weights = MobileNet_V2_Weights.DEFAULT
    _densenet_weights = DenseNet121_Weights.DEFAULT
except Exception:
    _mobilenet_weights = None
    _densenet_weights = None

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolutional Layer
    A = Adjacency Matrix (learned or static)
    X = Input Features
    Output = Activation(A * X * W)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (Batch, Nodes, In_Features)
        # adj: (Batch, Nodes, Nodes)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

class DRModel(nn.Module):
    def __init__(self, num_classes=5, backbone='mobilenet'):
        super(DRModel, self).__init__()
        
        # 1. Backbone: Feature Extractor
        if backbone == 'mobilenet':
            self.base_model = models.mobilenet_v2(weights=_mobilenet_weights)
            # MobileNetV2 features end at 1280 channels
            self.in_channels = 1280
            # Remove the classifier
            self.features = self.base_model.features
        elif backbone == 'densenet':
            self.base_model = models.densenet121(weights=_densenet_weights)
            self.in_channels = 1024
            self.features = self.base_model.features
        else:
            raise ValueError("Backbone must be 'mobilenet' or 'densenet'")

        # 2. Graph Convolutional Network Module
        # We will treat the spatial grid (H x W) as nodes in a graph
        # For MobileNetV2 input 224x224, output features are 7x7 = 49 nodes
        self.num_nodes = 7 * 7 
        self.gcn_in_dim = self.in_channels
        self.gcn_hid_dim = 512
        
        # Projection to interaction space (graph construction)
        self.conv_transform = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        
        # GCN Layer
        self.gcn = GraphConvolution(self.in_channels, self.in_channels)
        
        # 3. Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        
        # 1. Extract Features
        x = self.features(x) # [Batch, C, H, W] -> [Batch, 1280, 7, 7]
        
        b, c, h, w = x.size()
        
        # 2. GCN Branch
        # Construct Adjacency Matrix based on feature similarity
        # Reshape to [Batch, Nodes, Features] where Nodes = H*W
        x_flat = x.view(b, c, -1).permute(0, 2, 1) # [Batch, 49, 1280]
        
        # Compute similarity matrix (Adjacency)
        # A = softmax(X * X_T)
        scale = c ** -0.5
        adj = torch.bmm(x_flat, x_flat.transpose(1, 2)) * scale
        adj = F.softmax(adj, dim=-1) # [Batch, 49, 49]
        
        # Apply GCN
        gcn_out = self.gcn(x_flat, adj) # [Batch, 49, 1280]
        gcn_out = F.relu(gcn_out)
        
        # Reshape back to feature map
        gcn_out = gcn_out.permute(0, 2, 1).view(b, c, h, w)
        
        # Residual connection: Original features + GCN features
        x = x + gcn_out
        
        # 3. Classification
        x = self.global_pool(x) # [Batch, 1280, 1, 1]
        x = torch.flatten(x, 1) # [Batch, 1280]
        x = self.classifier(x)  # [Batch, 5]
        
        return x

def get_model(device='cpu', filepath=None):
    model = DRModel(backbone='mobilenet')
    if filepath:
        # Load weights if provided
        try:
            state_dict = torch.load(filepath, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Could not load weights: {e}")
            print("Using initialized model.")
    
    model.to(device)
    model.eval()
    return model
