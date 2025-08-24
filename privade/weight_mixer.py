import copy
import torch
import torch.nn as nn
from typing import Tuple, Union


class InvertibleLinear(nn.Module):
    """
    Invertible linear layer using orthogonal matrices.
    For use in linear layer mixing where we need exact invertibility.
    """
    def __init__(self, features: int, bias: bool = False):
        super().__init__()
        self.features = features
        
        # Initialize as orthogonal matrix (invertible)
        with torch.no_grad():
            Q, _ = torch.linalg.qr(torch.randn(features, features))
        
        self.weight = nn.Parameter(Q)
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        
    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def get_inverse_weight(self):
        """Get the inverse transformation matrix."""
        return self.weight.T  # For orthogonal matrices, inverse = transpose


def _get_last_layer(model: nn.Module) -> Tuple[str, nn.Module]:
    """Get the last layer of a model and its name."""
    last_name = None
    last_layer = None
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                last_name = name
                last_layer = module
    
    return last_name, last_layer


def _get_first_layer(model: nn.Module) -> Tuple[str, nn.Module]:
    """Get the first layer of a model and its name."""
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                return name, module
    
    return None, None


def _fold_matrix_into_conv(conv: nn.Conv2d, matrix: torch.Tensor):
    """
    Fold a transformation matrix into a Conv2d layer's weights.
    conv.weight: [out_c, in_c, kh, kw]
    We want W' such that W'(M @ x) == W @ x, so W' = W @ M^{-1}
    """
    with torch.no_grad():
        W = conv.weight.data
        out_c, in_c, kh, kw = W.shape
        assert matrix.shape == (in_c, in_c), f"Matrix shape {matrix.shape} must match in_channels {in_c}"
        
        # Reshape and apply transformation
        W_flat = W.permute(0, 2, 3, 1).reshape(-1, in_c)  # [(out*kh*kw), in]
        W_new_flat = W_flat @ matrix  # Apply transformation
        W_new = W_new_flat.reshape(out_c, kh, kw, in_c).permute(0, 3, 1, 2).contiguous()
        conv.weight.copy_(W_new)


def _fold_matrix_into_linear(linear: nn.Linear, matrix: torch.Tensor):
    """
    Fold a transformation matrix into a Linear layer's weights.
    linear.weight: [out_features, in_features]
    We want W' such that W'(M @ x) == W @ x, so W' = W @ M^{-1}
    """
    with torch.no_grad():
        W = linear.weight.data
        out_features, in_features = W.shape
        assert matrix.shape == (in_features, in_features), f"Matrix shape {matrix.shape} must match in_features {in_features}"
        
        W_new = W @ matrix
        linear.weight.copy_(W_new)


def weight_mixer(model_A: nn.Module, model_B: nn.Module) -> Tuple[nn.Module, nn.Module]:
    """
    Add weight mixing between two models at their boundary.
    
    Args:
        model_A: First model (will have mixer added at the end)
        model_B: Second model (will have unmixer folded into first layer)
    
    Returns:
        Tuple of (modified_model_A, modified_model_B)
    
    Raises:
        ValueError: If models don't have compatible boundary layers
    """
    model_A = model_A.to('cpu')
    model_B = model_B.to('cpu')
    # Get boundary layers
    last_name_A, last_layer_A = _get_last_layer(model_A)
    first_name_B, first_layer_B = _get_first_layer(model_B)
    
    if last_layer_A is None:
        raise ValueError("model_A has no Conv2d or Linear layers")
    if first_layer_B is None:
        raise ValueError("model_B has no Conv2d or Linear layers")
    
    # Check compatibility
    is_conv_case = isinstance(last_layer_A, nn.Conv2d) and isinstance(first_layer_B, nn.Conv2d)
    is_linear_case = isinstance(last_layer_A, nn.Linear) and isinstance(first_layer_B, nn.Linear)
    
    if not (is_conv_case or is_linear_case):
        raise ValueError(
            f"Incompatible boundary layers: model_A ends with {type(last_layer_A).__name__}, "
            f"model_B starts with {type(first_layer_B).__name__}. "
            f"Both must be Conv2d or both must be Linear."
        )
    
    # Create copies to avoid modifying originals
    model_A_copy = copy.deepcopy(model_A)
    model_B_copy = copy.deepcopy(model_B)
    
    if is_conv_case:
        # Conv2d case: use 1x1 convolution as mixer
        out_channels = last_layer_A.out_channels
        in_channels = first_layer_B.in_channels
        
        if out_channels != in_channels:
            raise ValueError(
                f"Channel mismatch: model_A outputs {out_channels} channels, "
                f"model_B expects {in_channels} channels"
            )
        
        # Create orthogonal mixing matrix
        with torch.no_grad():
            Q, _ = torch.linalg.qr(torch.randn(out_channels, out_channels))
        
        mixer = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            mixer.weight.copy_(Q.view(out_channels, out_channels, 1, 1))
        
        # Add mixer to model_A
        model_A_copy = nn.Sequential(model_A_copy, mixer)
        
        # Fold inverse into model_B's first conv layer
        first_layer_B_copy = None
        for name, module in model_B_copy.named_modules():
            if name == first_name_B:
                first_layer_B_copy = module
                break
        
        if first_layer_B_copy is not None:
            _fold_matrix_into_conv(first_layer_B_copy, Q.T)  # Q.T is inverse for orthogonal matrix
    
    else:
        # Linear case: use invertible linear layer as mixer
        out_features = last_layer_A.out_features
        in_features = first_layer_B.in_features
        
        if out_features != in_features:
            raise ValueError(
                f"Feature mismatch: model_A outputs {out_features} features, "
                f"model_B expects {in_features} features"
            )
        
        # Create invertible linear mixer
        mixer = InvertibleLinear(out_features, bias=False)
        
        # Add mixer to model_A
        model_A_copy = nn.Sequential(model_A_copy, mixer)
        
        # Fold inverse into model_B's first linear layer
        first_layer_B_copy = None
        for name, module in model_B_copy.named_modules():
            if name == first_name_B:
                first_layer_B_copy = module
                break
        
        if first_layer_B_copy is not None:
            inverse_weight = mixer.get_inverse_weight()
            _fold_matrix_into_linear(first_layer_B_copy, inverse_weight)
    
    return model_A_copy, model_B_copy
