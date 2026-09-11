import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import copy



class CNN_sup(nn.Module):
    def __init__(self, repr_dim=128, num_classes=10, in_ch=1, hd1=32, hd2=32, hd3=32, hw=28):
        super(CNN_sup, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=hd1, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=hd1, out_channels=hd2, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=hd2, out_channels=hd3, kernel_size=3, padding=1, stride=1)
        
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(hd3 * hw * hw, num_classes)

    def forward(self, x, supervised=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
    def forward(self, x):
        return self.model(x)



class FC_Supervised(nn.Module):
    def __init__(self, input_dim=28*28, hd1=24, hd2=26, output_dim=10):
        super(FC_Supervised, self).__init__()
        self.fc1 = nn.Linear(input_dim, hd1)
        self.fc2 = nn.Linear(hd1, hd2)
        self.fc3 = nn.Linear(hd2, output_dim)
    
    def forward(self, x, supervised=False):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x


class TinyTransformer(nn.Module):
    def __init__(self, patch_size=7, embed_dim=32, num_heads=2, ff_dim=64, num_classes=10):
        super(TinyTransformer, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2

        self.proj = nn.Linear(patch_size*patch_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, supervised=False):
        B = x.shape[0]
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, 1, -1, self.patch_size, self.patch_size)
        x = x.view(B, self.num_patches, self.patch_size * self.patch_size)
        x = self.proj(x)

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SSLModel(nn.Module):
    def __init__(self, backbone, projector):
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections

class BottleneckBlock(nn.Module):

    def __init__(self, width, bottleneck_factor=4):
        super().__init__()
        bottleneck_width = int(width / bottleneck_factor)
        self.fc1 = nn.Linear(width, bottleneck_width)
        self.fc2 = nn.Linear(bottleneck_width, width)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += identity
        out = self.activation(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)

        Q, K, V = self.query(x), self.key(x), self.value(x)

        scores = torch.bmm(Q, K.transpose(1, 2))
        attention_weights = self.softmax(scores / (K.size(-1) ** 0.5))
        
        context = torch.bmm(attention_weights, V)
        
        return context.squeeze(1)

class ResMLP(nn.Module):
    def __init__(self, in_features=110, width=1032, num_blocks=3, num_classes=1, num_heads=1):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, width)
        if num_heads<2:
            self.attention = SelfAttention(embed_dim=width)
        else:
            self.attention = MultiHeadSelfAttention(embed_dim=width, num_heads=num_heads)

        self.attention_norm = nn.LayerNorm(width) #####
        self.blocks = nn.ModuleList([BottleneckBlock(width) for _ in range(num_blocks)])
        
        self.final_layer = nn.Linear(width, num_classes)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.activation(x)

        identity = x #####
        x = self.attention_norm(x)  ####
        x = self.attention(x)  ####
        x = x + identity 
        
        for block in self.blocks:
            x = block(x)
            
        # x = self.final_layer(x)
        return x

class BYOL2(nn.Module):
    def __init__(self, backbone, projector, feature_dim, projector_out_dim):
        super().__init__()
        self.online_network = nn.Sequential(backbone, projector)
        
        self.predictor = nn.Sequential(
            nn.Linear(projector_out_dim, 2048), 
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, projector_out_dim) 
        ).to(next(backbone.parameters()).device)
        
        self.target_network = copy.deepcopy(self.online_network)
        
        for param in self.target_network.parameters():
            param.requires_grad = False
            
    def forward(self, x1, x2):

        online_proj1 = self.online_network(x1)
        online_proj2 = self.online_network(x2)
        
        online_pred1 = self.predictor(online_proj1)
        online_pred2 = self.predictor(online_proj2)
        
        with torch.no_grad():
            target_proj1 = self.target_network(x1)
            target_proj2 = self.target_network(x2)
            
        return online_pred1, online_pred2, target_proj1.detach(), target_proj2.detach()

    def update_target_network(self, tau):
        online_params = self.online_network.parameters()
        target_params = self.target_network.parameters()
        for o_param, t_param in zip(online_params, target_params):
            t_param.data.mul_(tau).add_(o_param.data, alpha=1 - tau)
    
    def get_backbone(self):
        return self.online_network[0]

class BYOL(nn.Module):
    def __init__(self, backbone, feature_dim=512):

        super().__init__()
        self.online_encoder = backbone
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            # nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(1024, feature_dim)
        ).to(next(backbone.parameters()).device)
        
        self.target_encoder = copy.deepcopy(backbone)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x1, x2):
    
        online_z1 = self.online_encoder(x1)
        online_z2 = self.online_encoder(x2)

        online_pred1 = self.predictor(online_z1)
        online_pred2 = self.predictor(online_z2)

        with torch.no_grad():
            target_z1 = self.target_encoder(x1)
            target_z2 = self.target_encoder(x2)
            
        return online_pred1, online_pred2, target_z1.detach(), target_z2.detach()

    def update_target_network(self, tau):
        """
        (EMA) update.
        """
        with torch.no_grad():
            online_params = self.online_encoder.parameters()
            target_params = self.target_encoder.parameters()

            for o_param, t_param in zip(online_params, target_params):
                t_param.data.mul_(tau).add_(o_param.data, alpha=1 - tau)