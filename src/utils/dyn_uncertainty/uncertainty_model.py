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
            x = F.dropout(x, p=0.2)

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




class AffineSlowNet(nn.Module):
    """Single-layer affine transform for slow uncertainty prior.
    384 -> 96 -> 1, with Softplus output.
    Much simpler than the Fast MLP; designed for smooth, generalisable priors.
    """
    def __init__(self, input_dim=384, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.softplus = nn.Softplus()

    def forward(self, x):
        H, W, C = x.shape[-3:]
        has_batch = (x.dim() == 4)
        if not has_batch:
            x = x.unsqueeze(0)
        B = x.shape[0]
        x = x.view(-1, C)
        x = F.relu(self.fc1(x))
        x = self.softplus(self.fc2(x))
        x = x.view(B, H, W) if has_batch else x.view(H, W)
        return x



class FirstPrinciplesUncertaintyModel(nn.Module):  
    def __init__(self, input_dim=384):
        super().__init__()
        self.net_fast = MLPNetwork(input_dim=input_dim, net_depth=2)

    def forward(self, x, dino_warp_scores=None, image_grad=None):
        u_fast = self.net_fast(x)
        return u_fast, None


class TrackerSlowNet(nn.Module):
    """Single-layer network for tracker-side uncertainty.
    Trained inside ba() using SCE NLL formulation.
    """
    def __init__(self, input_dim=384):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1),
            # nn.SiLU(),
            # nn.Linear(64, 1)
        )
        nn.init.kaiming_normal_(self.fc[0].weight,mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.fc[0].bias,mode='fan_in', nonlinearity='linear')
        # nn.init.xavier_uniform_(self.fc[2].weight)
        # nn.init.constant_(self.fc[2].bias, 0.0)

        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: [B, H, W, C] or [H, W, C]
        has_batch = (x.dim() == 4)
        if not has_batch:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        x = x.reshape(-1, C)
        x = self.fc(x)
        x = self.softplus(x)
        
        return x.view(B, H, W) if has_batch else x.view(H, W)


class LoRATrackerNet(nn.Module):
    """
    Tracker-side uncertainty network using Low-Rank Adaptation (LoRA).
    It wraps the mapper's base MLPNetwork and injects low-rank perturbations 
    during tracking to rapidly fit geometric motion without destroying the base prior.
    """
    def __init__(self, base_net: nn.Module, rank: int = 4):
        super().__init__()
        self.base_net = base_net
        self.rank = rank
        
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
            x = base_out + lora_out
            x = self.base_net.net_activation(x)
            # dropout is disabled during tracking due to eval() or just implicitly
        
        base_out = self.base_net.output_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_out_A), self.lora_out_B)
        x = base_out + lora_out
        x = self.softplus(x)

        return x.view(B, H, W) if has_batch else x.view(H, W)



'''
class EMAUncertaintyModel(nn.Module):  
    def __init__(self, input_dim=384, ema_momentum=0.999):
        super().__init__()
   
        self.net_fast = MLPNetwork(input_dim=input_dim, net_depth=2)
        self.net_slow = MLPNetwork(input_dim=input_dim, net_depth=2)
    
        for param in self.net_slow.parameters():
            param.requires_grad = True
        self.distill_loss = 0.0
        self.register_buffer("u_baseline", torch.tensor(0.15))
        
        # Track statistics for analytical CDF Matching (Z-Score Alignment)
        # self.register_buffer("mu_fast", torch.tensor(0.15))
        # self.register_buffer("var_fast", torch.tensor(0.01))
        self.register_buffer("mu_slow", torch.tensor(0.15))
        self.register_buffer("var_slow", torch.tensor(0.01))

    def forward(self, x, dino_warp_scores=None, image_grad=None):
       
        u_fast = self.net_fast(x)
        u_slow = self.net_slow(x)
        if self.training:
            if dino_warp_scores is not None and image_grad is not None:
                with torch.no_grad():
                    self.mu_slow.data = 0.99 * self.mu_slow.data + 0.01 * u_slow.mean()
                    self.var_slow.data = 0.99 * self.var_slow.data + 0.01 * u_slow.var(unbiased=False)

                H_dino, W_dino = x.shape[-3], x.shape[-2]
                
                # Pool DINOv2 warping Semantic Consistency Error (SCE) and image gradient
                M = F.adaptive_avg_pool2d(
                    dino_warp_scores.float().unsqueeze(0).unsqueeze(0) if dino_warp_scores.dim() == 2 else dino_warp_scores.float().unsqueeze(1),
                    (H_dino, W_dino)
                )
                M = M.squeeze(0).squeeze(0) if dino_warp_scores.dim() == 2 else M.squeeze(1)
                # M = diffuse_uncertainty(M, x, theta_sim=0.80)

                G = F.adaptive_avg_pool2d(
                    image_grad.float().unsqueeze(0) if image_grad.dim() == 3 else image_grad.float(),
                    (H_dino, W_dino)
                )
                G = G.squeeze(0).squeeze(0) if image_grad.dim() == 3 else G.squeeze(1)
                
                static_mask = (1.0 - M)
                if static_mask.sum() > 0:
                    current_static_u = (u_fast.detach() * static_mask).sum() / static_mask.sum()
                    self.u_baseline.data = 0.99 * self.u_baseline.data + 0.01 * current_static_u
                    
                W_dynamic = M
                dyn_floor = 0.5
                T_dynamic = u_fast.detach() + (M * dyn_floor) * (1 - u_fast.detach())    

                W_static_texture = (1.0 - M) * G
                T_static_texture = torch.full_like(u_fast, self.u_baseline.item())
                
                C_photo = torch.exp(-u_fast.detach())
                W_general = (1.0 - M) * (1.0 - G) * C_photo
                T_general = u_fast.detach()
                
                lambda_acquire = 1.0
                lambda_forget = 0.002
                alpha = 20.0  # Sharpness of transition

                u_slow_detached = u_slow.detach()

                delta_dyn = T_dynamic - u_slow_detached
                # lambda_dyn = lambda_forget + (lambda_acquire - lambda_forget) * torch.sigmoid(alpha * delta_dyn)
                lambda_dyn = 1.0
                distill_dynamic = W_dynamic * lambda_dyn * (u_slow - T_dynamic).pow(2)

                delta_stat = T_static_texture - u_slow_detached
                lambda_stat = lambda_forget + (lambda_acquire - lambda_forget) * torch.sigmoid(alpha * delta_stat)
                distill_static_texture = W_static_texture * lambda_stat * (u_slow - T_static_texture).pow(2)

                delta_gen = T_general - u_slow_detached
                lambda_gen = lambda_forget + (lambda_acquire - lambda_forget) * torch.sigmoid(alpha * delta_gen)
                distill_general = W_general * lambda_gen * (u_slow - T_general).pow(2)

                self.distill_loss = (distill_dynamic + distill_static_texture + distill_general).mean()
            else:
                weight = torch.exp(-u_fast.detach())
                self.distill_loss = (weight * (u_slow - u_fast.detach()).pow(2)).mean()
        else:
            self.distill_loss = 0.0
            
    
        # # Z-Space Contrast Stretching: Push high uncertainties higher
        # # In standardized Z-space, the noise floor (mean) is precisely 0.0.
        # std_slow = torch.sqrt(self.var_slow + 1e-6)
        # z_slow = (u_slow - self.mu_slow) / std_slow
        
        # # Only stretch confidently high uncertainties (z > 1.0). 
        # # This prevents negative z-values (low uncertainty) from becoming positive,
        # # and avoids shrinking/stretching values in the middle (-1.0 to 1.0).
        # z_slow = z_slow * torch.abs(z_slow)
        # u_slow = z_slow * std_slow + self.mu_slow
        
        return u_fast, u_slow, u_fast
'''




