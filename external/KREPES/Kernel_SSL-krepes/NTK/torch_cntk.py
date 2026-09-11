import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, jacrev, jvp, vjp
from functorch import make_functional_with_buffers
import inspect


class ConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, input_size_hw=(28, 28)):
        super().__init__()
        self.features  = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, *input_size_hw)
            flattened_size = self.features(dummy_input).flatten().shape[0]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, H, W]
        if x.shape[-1] == self.features[0].in_channels:
            x = x.permute(0, 3, 1, 2)
            
        x = self.features(x)
        x = self.classifier(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features=110, num_classes=2):
        super().__init__()
        activation_fn = nn.GELU()

        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            activation_fn,
            nn.Linear(512, 512),
            activation_fn,
            nn.Linear(512, 512),
            activation_fn,

            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += identity
        out = self.activation(out)
        
        return out
        
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
    def __init__(self, in_features=110, width=512, num_blocks=3, num_classes=2):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, width)
        self.attention = SelfAttention(embed_dim=width) #####
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
            
        x = self.final_layer(x)
        return x


# --- Helper Modules ---
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        b, n, _ = x.shape
        h, d = self.heads, self.dim_head
        qkv = self.to_qkv(x)                     
        q, k, v = qkv.chunk(3, dim=-1)             

        q = q.view(b, n, h, d).transpose(1, 2).contiguous()
        k = k.view(b, n, h, d).transpose(1, 2).contiguous()
        v = v.view(b, n, h, d).transpose(1, 2).contiguous()

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  

        out = out.transpose(1, 2).contiguous().view(b, n, h * d)
        return self.to_out(out)

class InterSampleAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    
    def forward(self, x):
     
        b, n, _ = x.shape
        h, d = self.heads, self.dim_head

 
        x_nb = x.transpose(0, 1).contiguous() 

        qkv = self.to_qkv(x_nb)                
        q, k, v = qkv.chunk(3, dim=-1)       

       
        q = q.view(n, b, h, d).transpose(1, 2).contiguous()
        k = k.view(n, b, h, d).transpose(1, 2).contiguous()
        v = v.view(n, b, h, d).transpose(1, 2).contiguous()

     
        sim = torch.einsum('n h i d, n h j d -> n h i j', q, k) * self.scale  
        attn = sim.softmax(dim=-1)
        out = torch.einsum('n h i j, n h j d -> n h i d', attn, v)       

    
        out = out.transpose(1, 2).contiguous().view(n, b, h * d)
        out = self.to_out(out)              
        return out.transpose(0, 1).contiguous()
        
class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_classes=1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_categories = len(categories)
        self.num_continuous = num_continuous

        self.categorical_embedders = nn.ModuleList([
            nn.Embedding(num_categories, dim) for num_categories in categories
        ])

        self.continuous_norm = nn.LayerNorm(num_continuous) if num_continuous > 0 else None
        self.continuous_embedders = nn.ModuleList([
            nn.Linear(1, dim) for _ in range(num_continuous)
        ]) if num_continuous > 0 else None
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, InterSampleAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout)))
            ]))


        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x_categ, x_cont):
        embeds = []
        if self.num_categories > 0:
            for i, emb in enumerate(self.categorical_embedders):
                embeds.append(emb(x_categ[:, i]).unsqueeze(1))

        if self.num_continuous > 0:
            x_cont_norm = self.continuous_norm(x_cont)
            for i, emb in enumerate(self.continuous_embedders):
                embeds.append(emb(x_cont_norm[:, i].unsqueeze(-1)).unsqueeze(1))
        
        x = torch.cat(embeds, dim=1)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for self_attn, inter_sample_attn, ff in self.layers:
            x = self_attn(x)
            x = inter_sample_attn(x)
            x = ff(x)
            
        cls_token_output = x[:, 0]
        return self.head(cls_token_output)
        
class MNISTConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
    
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 28, 28)
            flattened_size = self.features(dummy_input).flatten().shape[0]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def compute_ntk(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor, full_ntk=False):

    model.eval()
    params = {name: p for name, p in model.named_parameters()}

    def get_flat_jacobian(x_sample: torch.Tensor):

        def model_fn(params_dict):
            return functional_call(model, params_dict, (x_sample.unsqueeze(0),))
        jac = jacrev(model_fn)(params)
        flat_jac = torch.cat([j.flatten(1) for j in jac.values()], dim=1)
        
        return flat_jac

    vmapped_get_jacobian = vmap(get_flat_jacobian)

    jac1 = vmapped_get_jacobian(x1)  
    jac2 = vmapped_get_jacobian(x2)  


    if full_ntk:
        ntk = torch.einsum('ncp, mdp -> nmdc', jac1, jac2)
    else:
        ntk = torch.einsum('ncp, mcp -> nm', jac1, jac2)
        
    return ntk

    
def compute_ntk_vps(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor, compute='trace', chunk_size=None):
    model.eval()
    fmodel, params, buffers = make_functional_with_buffers(model)


    def fnet_single(params, x_sample):
        
        return fmodel(params, buffers, x_sample.unsqueeze(0)).squeeze(0)

    def get_ntk_for_single_pair(x1_single, x2_single):
        def func_x1(p):
            return fnet_single(p, x1_single)

        def func_x2(p):
            return fnet_single(p, x2_single)

        output_x2, vjp_fn_x2 = vjp(func_x2, params)

        def get_ntk_slice(vec):
            vjp_val = vjp_fn_x2(vec)
            _, jvp_val = jvp(func_x1, (params,), vjp_val)
            return jvp_val
            
        identity_matrix = torch.eye(output_x2.numel(), device=output_x2.device, dtype=output_x2.dtype)
        return vmap(get_ntk_slice, chunk_size=chunk_size)(identity_matrix)

    result = vmap(vmap(get_ntk_for_single_pair, in_dims=(None, 0)), in_dims=(0, None))(x1, x2)

    if compute == 'full':
        return result
    elif compute == 'trace':
        return torch.einsum('NMKK->NM', result)
    elif compute == 'diagonal':
        return torch.einsum('NMKK->NMK', result)
    else:
        raise ValueError(f"Unknown compute option: {compute}. Options are 'full', 'trace', 'diagonal'.")