import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.dyn_uncertainty.mapping_utils import compute_dino_regularization_loss


class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, output_dim: int = 1, 
                 net_depth: int = 2, net_activation=F.relu, weight_init: str = 'he_uniform', decrease_dim=False):
        super(MLPNetwork, self).__init__()
        
        # Initialize MLP layers
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for i in range(net_depth):
            if decrease_dim:
                out_dim = hidden_dim
            else:
                out_dim = hidden_dim
            dense_layer = nn.Linear(current_dim, out_dim)
            
            # Apply weight initialization
            if weight_init == 'he_uniform':
                nn.init.kaiming_uniform_(dense_layer.weight, nonlinearity='relu')
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(dense_layer.weight)
            else:
                raise NotImplementedError(f"Unknown Weight initialization method {weight_init}")

            self.layers.append(dense_layer)
            current_dim = out_dim
        
        self.output_layer_input_dim = current_dim
        # Initialize output layer
        self.output_layer = nn.Linear(self.output_layer_input_dim, output_dim)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        
        # Set activation function
        self.net_activation = net_activation
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get input dimensions
        H, W, C = x.shape[-3:]
        input_with_batch_dim = True
        
        # Add batch dimension if not present
        if len(x.shape) == 3:
            input_with_batch_dim = False
            x = x.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # Flatten input for MLP
        x = x.view(-1, x.size()[-1])
        
        # Pass through MLP layers
        for layer in self.layers:
            x = layer(x)
            x = self.net_activation(x)

        # Pass through output layer and apply softplus activation
        x = self.output_layer(x)
        x = self.softplus(x)

        # Reshape output to original dimensions
        if input_with_batch_dim:
            x = x.view(batch_size, H, W)
        else:
            x = x.view(H, W)

        return x

def generate_uncertainty_mlp(n_features: int):
    # Create and return the shared-backbone uncertainty model.
    # network = EMAUncertaintyModel(input_dim=n_features).cuda()
    network = FirstPrinciplesUncertaintyModel(input_dim=n_features).cuda()
    return network






class FirstPrinciplesUncertaintyModel(nn.Module):  
    def __init__(self, input_dim=384):
        super().__init__()
        self.net_fast = MLPNetwork(input_dim=input_dim, net_depth=2)

    def forward(self, x):
        u_fast = self.net_fast(x)
        return u_fast, None





class LoRATrackerNet(nn.Module):
    """
    Tracker-side uncertainty network using Low-Rank Adaptation (LoRA).
    It wraps the mapper's base MLPNetwork and injects low-rank perturbations 
    during tracking to rapidly fit geometric motion without destroying the base prior.
    """
    def __init__(self, base_net: nn.Module, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base_net = base_net
        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        
        # base_net is an MLPNetwork
        for layer in self.base_net.layers:
            self.lora_A.append(nn.Parameter(torch.zeros(rank, layer.in_features)))
            self.lora_B.append(nn.Parameter(torch.zeros(layer.out_features, rank)))
            
        self.lora_out_A = nn.Parameter(torch.zeros(rank, self.base_net.output_layer.in_features))
        self.lora_out_B = nn.Parameter(torch.zeros(self.base_net.output_layer.out_features, rank))
        
        for a in self.lora_A:
            nn.init.kaiming_normal_(a, mode='fan_in', nonlinearity='linear')
        for b in self.lora_B:
            nn.init.zeros_(b)
            
        nn.init.kaiming_normal_(self.lora_out_A, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.lora_out_B)
        
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_batch = (x.dim() == 4)
        if not has_batch:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        x = x.reshape(-1, C)
        
        for i, layer in enumerate(self.base_net.layers):
            base_out = layer(x)
            lora_out = F.linear(F.linear(x, self.lora_A[i]), self.lora_B[i])
            x = base_out + lora_out * self.scaling
            x = self.base_net.net_activation(x)
            # dropout is disabled during tracking due to eval() or just implicitly
        
        base_out = self.base_net.output_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_out_A), self.lora_out_B)
        x = base_out + lora_out * self.scaling
        x = self.softplus(x)

        return x.view(B, H, W) if has_batch else x.view(H, W)








