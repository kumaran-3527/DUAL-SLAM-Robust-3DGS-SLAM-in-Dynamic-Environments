import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, output_dim: int = 1, 
                 net_depth: int = 2, net_activation=F.relu, weight_init: str = 'he_uniform'):
        super(MLPNetwork, self).__init__()
        
        self.output_layer_input_dim = hidden_dim
        
        # Initialize MLP layers
        self.layers = nn.ModuleList()
        for i in range(net_depth):
            dense_layer = nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            
            # Apply weight initialization
            if weight_init == 'he_uniform':
                nn.init.kaiming_uniform_(dense_layer.weight, nonlinearity='relu')
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(dense_layer.weight)
            else:
                raise NotImplementedError(f"Unknown Weight initialization method {weight_init}")

            self.layers.append(dense_layer)
        
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
    # Create and return an MLP network with the specified input dimensions
    network = EMAUncertaintyModel(input_dim=n_features).cuda()
    return network


class EMAUncertaintyModel(nn.Module):
    def __init__(self, input_dim=384, ema_momentum=0.999):
        super().__init__()
        self.net_fast = MLPNetwork(input_dim=input_dim)
        self.net_slow = MLPNetwork(input_dim=input_dim)
        self.net_slow.load_state_dict(self.net_fast.state_dict())
        for param in self.net_slow.parameters():
            param.requires_grad = True
        self.distill_loss = 0.0
        self.register_buffer("u_baseline", torch.tensor(0.15))
        
        # Track statistics for analytical CDF Matching (Z-Score Alignment)
        self.register_buffer("mu_fast", torch.tensor(0.15))
        self.register_buffer("var_fast", torch.tensor(0.01))
        self.register_buffer("mu_slow", torch.tensor(0.15))
        self.register_buffer("var_slow", torch.tensor(0.01))

    def forward(self, x, dino_warp_scores=None, image_grad=None):
        u_fast = self.net_fast(x)
        u_slow = self.net_slow(x)
        if self.training:
            if dino_warp_scores is not None and image_grad is not None:
                with torch.no_grad():
                    self.mu_fast.data = 0.99 * self.mu_fast.data + 0.01 * u_fast.mean()
                    self.var_fast.data = 0.99 * self.var_fast.data + 0.01 * u_fast.var(unbiased=False)
                    self.mu_slow.data = 0.99 * self.mu_slow.data + 0.01 * u_slow.mean()
                    self.var_slow.data = 0.99 * self.var_slow.data + 0.01 * u_slow.var(unbiased=False)

                H_dino, W_dino = x.shape[-3], x.shape[-2]
                
                # Pool DINOv2 warping Semantic Consistency Error (SCE) and image gradient
                M = F.adaptive_avg_pool2d(
                    dino_warp_scores.float().unsqueeze(0).unsqueeze(0) if dino_warp_scores.dim() == 2 else dino_warp_scores.float().unsqueeze(1),
                    (H_dino, W_dino)
                )
                M = M.squeeze(0).squeeze(0) if dino_warp_scores.dim() == 2 else M.squeeze(1)
                
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
                T_dynamic = u_fast.detach()
                
                W_static_texture = (1.0 - M) * G
                T_static_texture = torch.full_like(u_fast, self.u_baseline.item())
                
                C_photo = torch.exp(-u_fast.detach())
                W_general = (1.0 - M) * (1.0 - G) * C_photo
                T_general = u_fast.detach()
                
                lambda_acquire = 1.0
                lambda_forget = 0.002

                u_slow_detached = u_slow.detach()
                # lambda_forget_dynamic = lambda_forget * torch.exp(-5.0 * torch.clamp(u_slow_detached - self.u_baseline.data, min=0.0))

                lambda_dyn = torch.where(T_dynamic > u_slow_detached, lambda_acquire, lambda_forget)
                distill_dynamic = W_dynamic * lambda_dyn * (u_slow - T_dynamic).pow(2)

                lambda_stat = torch.where(T_static_texture > u_slow_detached, lambda_acquire, lambda_forget)
                distill_static_texture = W_static_texture * lambda_stat * (u_slow - T_static_texture).pow(2)

                lambda_gen = torch.where(T_general > u_slow_detached, lambda_acquire, lambda_forget)
                distill_general = W_general * lambda_gen * (u_slow - T_general).pow(2)

                self.distill_loss = (distill_dynamic + distill_static_texture + distill_general).mean()
            else:
                weight = torch.exp(-u_fast.detach())
                self.distill_loss = (weight * (u_slow - u_fast.detach()).pow(2)).mean()
        else:
            self.distill_loss = 0.0
            
        # Analytical CDF Matching (Z-Score Alignment)
        std_fast = torch.sqrt(self.var_fast + 1e-6)
        std_slow = torch.sqrt(self.var_slow + 1e-6)
        
        z_fast = (u_fast - self.mu_fast) / std_fast
        z_slow = (u_slow - self.mu_slow) / std_slow
        
        # Take the max in the strictly normalized Z-space
        z_max = torch.max(z_fast, z_slow)
        
        # Project back to the Fast MLP's scale (which is the scale expected by the NLL mapping loss)
        u_aligned_max = z_max * std_fast + self.mu_fast
        
        # Ensure the output respects positivity and is never lower than the raw photometric error
        u_aligned_max = torch.max(u_fast, u_aligned_max)
        
        # Contrast Stretching: Push high uncertainties higher, low uncertainties lower
        # We use the global noise floor (mu_fast) as the pivot point.
        pivot = self.mu_fast.item() + 1e-6
        
        # Calculate relative uncertainty (u / pivot)
        rel_u = u_aligned_max / pivot
        
        # Square the relative uncertainty. 
        # Since x^2 shrinks values < 1 and exponentially grows values > 1,
        # this naturally pushes low uncertainties lower and high uncertainties MUCH higher.
        rel_u_stretched = rel_u ** 2.0
        
        # Map back to the absolute scale
        u_aligned_max = rel_u_stretched * pivot
        
        # Clamp to ensure valid variance boundaries
        u_aligned_max = torch.clamp(u_aligned_max, min=0.1)
        
        return u_fast, u_slow, u_aligned_max


