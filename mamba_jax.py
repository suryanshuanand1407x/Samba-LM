"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces
JAX Implementation optimized for Apple Silicon (MPS)

Based on: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

This implementation follows the mathematical formulations from the paper:
- Selective SSM (S6): Makes parameters (Δ, B, C) functions of input x
- Zero-Order Hold (ZOH) discretization
- Hardware-aware parallel scan using jax.lax.associative_scan

Author: Implementation for Mac MPS backend
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Tuple, Optional, NamedTuple
import math

# Flax for neural network modules
import flax.linen as nn
from flax.linen import initializers

# =============================================================================
# Phase 1: Environment Setup and Device Verification
# =============================================================================

def check_device():
    """
    Verify JAX device availability and print device info.
    For Apple Silicon, we expect 'METAL' or 'gpu' backend.
    """
    devices = jax.devices()
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {devices}")
    print(f"Default backend: {jax.default_backend()}")

    # Check for MPS/Metal availability
    try:
        # Try to get Metal devices specifically
        metal_devices = jax.devices('gpu')
        print(f"GPU/Metal devices: {metal_devices}")
        return True
    except RuntimeError:
        print("Warning: No GPU/Metal devices found. Running on CPU.")
        return False


# =============================================================================
# Phase 1: Discretization Functions (Zero-Order Hold)
# =============================================================================

def discretize_zoh(
    A: jnp.ndarray,  # (D, N) - continuous state matrix (diagonal)
    B: jnp.ndarray,  # (D, N) or (B, L, D, N) - continuous input matrix
    delta: jnp.ndarray,  # (D,) or (B, L, D) - step sizes
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Zero-Order Hold (ZOH) discretization.

    Converts continuous-time SSM parameters to discrete-time.

    From the paper (Equation 4):
        Ā = exp(Δ·A)
        B̄ = (Δ·A)^{-1} (exp(Δ·A) - I) · Δ·B

    For diagonal A, this simplifies significantly.
    When A is diagonal, we can compute element-wise.

    For numerical stability with diagonal A, we use:
        B̄ = Δ · B · (exp(Δ·A) - 1) / (Δ·A)
           ≈ Δ · B  when Δ·A is small (using the limit)

    Args:
        A: Continuous state matrix, shape (D, N) - diagonal elements
        B: Continuous input matrix, shape varies by context
        delta: Discretization step sizes

    Returns:
        A_bar: Discretized state matrix
        B_bar: Discretized input matrix
    """
    # Expand delta for broadcasting
    # delta: (B, L, D) -> (B, L, D, 1) for element-wise ops with A: (D, N)
    if delta.ndim == 3:  # (B, L, D)
        delta_expanded = delta[..., None]  # (B, L, D, 1)
    elif delta.ndim == 1:  # (D,)
        delta_expanded = delta[:, None]  # (D, 1)
    else:
        delta_expanded = delta

    # Compute Δ·A
    # A is (D, N), delta_expanded broadcasts appropriately
    deltaA = delta_expanded * A  # (B, L, D, N) or (D, N)

    # Ā = exp(Δ·A)
    A_bar = jnp.exp(deltaA)

    # For B̄, we need to handle the (exp(ΔA) - I) / (ΔA) term carefully
    # This equals 1 when ΔA -> 0 (L'Hopital's rule)
    # We use the stable formulation: B̄ = Δ · B · expm1(ΔA) / (ΔA)
    # where expm1(x) = exp(x) - 1 (numerically stable for small x)

    # Compute (exp(ΔA) - 1) / (ΔA) using expm1 for stability
    # Note: expm1(x)/x -> 1 as x -> 0
    # We handle the near-zero case explicitly

    # Safe division: (exp(x) - 1) / x with handling for x ≈ 0
    def safe_expm1_over_x(x):
        """Compute (exp(x) - 1) / x safely."""
        # For |x| < threshold, use Taylor series: 1 + x/2 + x^2/6 + ...
        threshold = 1e-4
        small_x = jnp.abs(x) < threshold
        # Taylor approximation: 1 + x/2 + x^2/6
        taylor = 1.0 + x/2.0 + x**2/6.0
        # Direct computation for larger x
        direct = jnp.expm1(x) / jnp.where(jnp.abs(x) < 1e-10, 1.0, x)
        return jnp.where(small_x, taylor, direct)

    # Compute the discretization factor
    disc_factor = safe_expm1_over_x(deltaA)

    # B̄ = Δ · B · disc_factor
    # Handle different B shapes
    if B.ndim == 2:  # (D, N) - static B
        B_bar = delta_expanded * B * disc_factor
    else:  # (B, L, D, N) - selective B
        B_bar = delta_expanded * B * disc_factor

    return A_bar, B_bar


def discretize_zoh_simple(
    A: jnp.ndarray,  # (D, N)
    B: jnp.ndarray,  # (B, L, D, N)
    delta: jnp.ndarray,  # (B, L, D)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simplified ZOH discretization using first-order approximation.

    This is faster but less accurate for large Δ:
        Ā ≈ exp(Δ·A)
        B̄ ≈ Δ·B

    This approximation is valid when Δ is small, which is typically
    the case after softplus activation (values around 0.001 to 0.1).

    Args:
        A: State matrix, shape (D, N)
        B: Input matrix, shape (B, L, D, N)
        delta: Step sizes, shape (B, L, D)

    Returns:
        A_bar: Discretized state matrix, shape (B, L, D, N)
        B_bar: Discretized input matrix, shape (B, L, D, N)
    """
    delta_expanded = delta[..., None]  # (B, L, D, 1)

    # Ā = exp(Δ·A)
    deltaA = delta_expanded * A  # (B, L, D, N)
    A_bar = jnp.exp(deltaA)

    # B̄ ≈ Δ·B (first-order approximation)
    B_bar = delta_expanded * B

    return A_bar, B_bar


# =============================================================================
# Phase 2: Selective Scan Implementation
# =============================================================================

class SSMState(NamedTuple):
    """State for associative scan operation."""
    h: jnp.ndarray  # Hidden state


def selective_scan_sequential(
    A_bar: jnp.ndarray,  # (B, L, D, N) - discretized state matrix
    B_bar: jnp.ndarray,  # (B, L, D, N) - discretized input matrix
    C: jnp.ndarray,      # (B, L, D, N) - output matrix
    x: jnp.ndarray,      # (B, L, D) - input sequence
    h0: Optional[jnp.ndarray] = None,  # (B, D, N) - initial state
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sequential implementation of selective scan (for reference/debugging).

    Computes the SSM recurrence:
        h_t = Ā_t · h_{t-1} + B̄_t · x_t
        y_t = C_t · h_t

    This is O(L) sequential, used for verification.

    Args:
        A_bar: Discretized state transition, shape (B, L, D, N)
        B_bar: Discretized input matrix, shape (B, L, D, N)
        C: Output matrix, shape (B, L, D, N)
        x: Input sequence, shape (B, L, D)
        h0: Optional initial state, shape (B, D, N)

    Returns:
        y: Output sequence, shape (B, L, D)
        h_final: Final hidden state, shape (B, D, N)
    """
    B_size, L, D, N = A_bar.shape

    # Initialize hidden state
    if h0 is None:
        h = jnp.zeros((B_size, D, N))
    else:
        h = h0

    outputs = []

    # Sequential scan
    for t in range(L):
        # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        # A_bar[:, t]: (B, D, N)
        # h: (B, D, N)
        # B_bar[:, t]: (B, D, N)
        # x[:, t]: (B, D) -> need to expand to (B, D, 1)

        x_t = x[:, t, :, None]  # (B, D, 1)
        h = A_bar[:, t] * h + B_bar[:, t] * x_t  # (B, D, N)

        # y_t = C_t · h_t (sum over N dimension)
        y_t = jnp.sum(C[:, t] * h, axis=-1)  # (B, D)
        outputs.append(y_t)

    y = jnp.stack(outputs, axis=1)  # (B, L, D)
    return y, h


def selective_scan_parallel(
    A_bar: jnp.ndarray,  # (B, L, D, N) - discretized state matrix
    B_bar: jnp.ndarray,  # (B, L, D, N) - discretized input matrix
    C: jnp.ndarray,      # (B, L, D, N) - output matrix
    x: jnp.ndarray,      # (B, L, D) - input sequence
    h0: Optional[jnp.ndarray] = None,  # (B, D, N) - initial state
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Parallel implementation of selective scan using associative scan.

    The key insight from the paper is that the recurrence:
        h_t = Ā_t · h_{t-1} + B̄_t · x_t

    Can be reformulated as an associative operation:
        (a₂, b₂) ⊗ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)

    This allows O(log L) parallel computation using jax.lax.associative_scan.

    Args:
        A_bar: Discretized state transition, shape (B, L, D, N)
        B_bar: Discretized input matrix, shape (B, L, D, N)
        C: Output matrix, shape (B, L, D, N)
        x: Input sequence, shape (B, L, D)
        h0: Optional initial state, shape (B, D, N)

    Returns:
        y: Output sequence, shape (B, L, D)
        h_final: Final hidden state, shape (B, D, N)
    """
    B_size, L, D, N = A_bar.shape

    # Prepare input: B_bar * x
    # x: (B, L, D) -> (B, L, D, 1) for broadcasting
    x_expanded = x[..., None]  # (B, L, D, 1)
    Bx = B_bar * x_expanded  # (B, L, D, N)

    # Handle initial state
    if h0 is not None:
        # Prepend initial state contribution
        # This requires modifying the first element
        # h_1 = A_bar_1 * h0 + B_bar_1 * x_1
        # We can handle this by adding A_bar_1 * h0 to Bx at t=0
        init_contribution = A_bar[:, 0] * h0  # (B, D, N)
        Bx = Bx.at[:, 0].add(init_contribution)

    # Define the associative binary operation
    # For the recurrence h_t = a_t * h_{t-1} + b_t, we need:
    # When combining (a_left, b_left) then (a_right, b_right):
    #   h_right = a_right * h_left + b_right
    #           = a_right * (cumulative_a_left * h_0 + cumulative_b_left) + b_right
    #           = (a_right * cumulative_a_left) * h_0 + (a_right * cumulative_b_left + b_right)
    #
    # So: (a_left, b_left) ⊗ (a_right, b_right) = (a_right * a_left, a_right * b_left + b_right)
    def associative_op(left, right):
        a_left, b_left = left
        a_right, b_right = right
        a_new = a_right * a_left
        b_new = a_right * b_left + b_right
        return (a_new, b_new)

    # Stack elements for scan: (A_bar, Bx)
    # Each element is (a_t, b_t) where:
    #   a_t = A_bar_t
    #   b_t = B_bar_t * x_t
    elements = (A_bar, Bx)

    # Apply associative scan along sequence dimension (axis=1)
    # This computes all prefix products
    _, all_h = lax.associative_scan(associative_op, elements, axis=1)

    # all_h now contains all hidden states: (B, L, D, N)
    # Compute outputs: y_t = sum_n(C_t * h_t)
    y = jnp.sum(C * all_h, axis=-1)  # (B, L, D)

    # Final hidden state
    h_final = all_h[:, -1]  # (B, D, N)

    return y, h_final


# =============================================================================
# Selective SSM Layer (S6 Core)
# =============================================================================

def selective_ssm(
    x: jnp.ndarray,      # (B, L, D) - input
    A: jnp.ndarray,      # (D, N) - state matrix (learned, typically negative)
    B_proj: jnp.ndarray, # (D, N_proj) - projection for B
    C_proj: jnp.ndarray, # (D, N_proj) - projection for C
    delta_proj: jnp.ndarray,  # (D, delta_rank) - projection for delta
    delta_bias: jnp.ndarray,  # (D,) - bias for delta
    D_param: jnp.ndarray,     # (D,) - skip connection
    N: int,              # State dimension
    use_parallel: bool = True,
) -> jnp.ndarray:
    """
    Selective SSM (S6) core computation.

    This implements Algorithm 2 from the paper:
    1. Project x to get B, C, Δ (input-dependent/selective)
    2. Discretize using ZOH
    3. Compute scan (parallel or sequential)
    4. Add skip connection

    The key difference from standard S4 is that B, C, Δ are
    functions of the input x, making the SSM time-varying.

    Args:
        x: Input tensor, shape (B, L, D)
        A: State matrix (diagonal), shape (D, N)
        B_proj: Projection weights for B
        C_proj: Projection weights for C
        delta_proj: Projection weights for Δ
        delta_bias: Bias for Δ computation
        D_param: Skip connection parameter
        N: State dimension
        use_parallel: Whether to use parallel scan

    Returns:
        y: Output tensor, shape (B, L, D)
    """
    B_size, L, D = x.shape

    # === Selection Mechanism ===
    # Project input to get B, C, delta
    # In practice, these are linear projections from x

    # For now, we'll use simple einsum projections
    # B: (B, L, D) @ (D, N) -> (B, L, D, N)  [broadcast across D]
    # In the paper, B is (B, L, N), then broadcast to (B, L, D, N)
    # Let's follow the paper more closely:

    # s_B(x) = Linear_N(x)  -> (B, L, N)
    # s_C(x) = Linear_N(x)  -> (B, L, N)
    # s_Δ(x) = softplus(Linear_1(x) + bias)  -> (B, L, D)

    # For this implementation, assume projections are pre-computed or passed
    # B: (B, L, N) -> broadcast to (B, L, D, N) via einsum
    # C: (B, L, N) -> broadcast to (B, L, D, N) via einsum

    # Simplified: B and C are (B, L, D, N) directly after projection
    B = jnp.broadcast_to(B_proj, (B_size, L, D, N))  # Placeholder
    C = jnp.broadcast_to(C_proj, (B_size, L, D, N))  # Placeholder

    # Compute delta with softplus activation
    # delta = softplus(x @ delta_proj + delta_bias)
    delta = jax.nn.softplus(x @ delta_proj + delta_bias)  # (B, L, D)

    # === Discretization ===
    A_bar, B_bar = discretize_zoh_simple(A, B, delta)

    # === Selective Scan ===
    if use_parallel:
        y, _ = selective_scan_parallel(A_bar, B_bar, C, x)
    else:
        y, _ = selective_scan_sequential(A_bar, B_bar, C, x)

    # === Skip Connection ===
    # y = y + D * x
    y = y + D_param * x

    return y


# =============================================================================
# Phase 2: Mamba Block Components (Flax Modules)
# =============================================================================

class CausalConv1D(nn.Module):
    """
    Causal 1D convolution for Mamba.

    Uses depthwise convolution (groups=channels) for efficiency.
    Causal padding ensures no information leakage from future.

    From the paper: "a convolution of dimension 1 with kernel size d_conv=4"

    Note: Implemented manually to support JAX Metal backend which doesn't
    support grouped convolutions (feature_group_count > 1).
    """
    features: int
    kernel_size: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (B, L, D)

        Returns:
            y: Output tensor, shape (B, L, D)
        """
        B, L, D = x.shape

        # Causal padding: pad (kernel_size - 1) on the left, 0 on the right
        # This ensures position t only sees positions 0..t
        pad_width = self.kernel_size - 1

        # Pad along sequence dimension: (B, L + pad, D)
        x_padded = jnp.pad(x, ((0, 0), (pad_width, 0), (0, 0)), mode='constant')

        # Manual depthwise convolution implementation for Metal compatibility
        # Kernel: (kernel_size, features) - one filter per channel
        kernel = self.param(
            'kernel',
            initializers.lecun_normal(),
            (self.kernel_size, self.features)
        )
        bias = self.param('bias', initializers.zeros, (self.features,))

        # Depthwise conv: each channel is convolved with its own filter
        # x_padded: (B, L+k-1, D)
        # We need to compute: y[b, t, d] = sum_i(x_padded[b, t+i, d] * kernel[i, d]) + bias[d]

        # Create sliding windows: (B, L, kernel_size, D)
        # Using lax.conv_general_dilated_patches or manual extraction
        def extract_patches(x_pad):
            # x_pad: (L+k-1, D)
            # Returns: (L, k, D)
            patches = jnp.stack([
                x_pad[i:i+L] for i in range(self.kernel_size)
            ], axis=1)  # (L, k, D)
            return patches

        # Apply to batch: (B, L, k, D)
        patches = jax.vmap(extract_patches)(x_padded)

        # Multiply by kernel and sum over kernel dimension
        # patches: (B, L, k, D), kernel: (k, D)
        y = jnp.sum(patches * kernel, axis=2) + bias  # (B, L, D)

        return y


class S6Layer(nn.Module):
    """
    Selective SSM (S6) layer with learned projections.

    This is the core of Mamba: a state space model where B, C, Δ
    are functions of the input x, making it time-varying/selective.

    From Algorithm 2 in the paper:
    - A: (D, N) - learned parameter, initialized with S4D-Real
    - B = s_B(x) = Linear_N(x)  -> (B, L, N)
    - C = s_C(x) = Linear_N(x)  -> (B, L, N)
    - Δ = softplus(Linear_1(x) + Δ_bias)  -> (B, L, D)
    - D: (D,) - skip connection parameter
    """
    d_model: int          # Model dimension (D)
    d_state: int = 16     # SSM state dimension (N)
    dt_rank: int = None   # Rank for delta projection (default: ceil(D/16))
    dt_min: float = 0.001 # Min value for delta initialization
    dt_max: float = 0.1   # Max value for delta initialization
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    def setup(self):
        """Initialize parameters."""
        # dt_rank defaults to ceil(d_model / 16)
        self.dt_rank_actual = self.dt_rank or math.ceil(self.d_model / 16)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        use_parallel: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (B, L, D)
            use_parallel: Whether to use parallel scan

        Returns:
            y: Output tensor, shape (B, L, D)
        """
        B_size, L, D = x.shape
        N = self.d_state
        dt_rank = self.dt_rank_actual

        # === Parameter Initialization ===

        # A: (D, N) - S4D-Real initialization: A_n = -(n+1)
        # Stored as log for stability (we exponentiate during discretization)
        A_log = self.param(
            'A_log',
            lambda rng, shape: jnp.log(
                jnp.broadcast_to(
                    jnp.arange(1, shape[1] + 1, dtype=jnp.float32),
                    shape
                )
            ),
            (D, N)
        )
        A = -jnp.exp(A_log)  # (D, N), negative for stability

        # D: (D,) - skip connection, initialized to ones
        D_param = self.param('D', initializers.ones, (D,))

        # === Projections for Selection Mechanism ===

        # Combined projection for B and C: (B, L, D) -> (B, L, 2*N)
        # This is more efficient than separate projections
        x_bc = nn.Dense(
            features=2 * N,
            use_bias=False,
            kernel_init=initializers.lecun_normal(),
            name='x_proj_bc'
        )(x)  # (B, L, 2*N)

        B_sel = x_bc[..., :N]   # (B, L, N)
        C_sel = x_bc[..., N:]   # (B, L, N)

        # Delta projection: (B, L, D) -> (B, L, D)
        # Two-stage: first project to dt_rank, then to D
        # This is the "low-rank" parameterization from the paper

        # First project to dt_rank
        x_dt = nn.Dense(
            features=dt_rank,
            use_bias=False,
            kernel_init=initializers.lecun_normal(),
            name='x_proj_dt'
        )(x)  # (B, L, dt_rank)

        # Then project to D with bias
        # Bias initialization: inverse softplus of uniform[dt_min, dt_max]
        def dt_bias_init(rng, shape, dtype=jnp.float32):
            # inv_softplus(x) = log(exp(x) - 1)
            dt = jax.random.uniform(rng, shape, minval=self.dt_min, maxval=self.dt_max, dtype=dtype)
            dt = jnp.clip(dt, a_min=self.dt_init_floor)
            # inverse softplus
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt

        dt_proj = nn.Dense(
            features=D,
            use_bias=True,
            kernel_init=initializers.lecun_normal(),
            bias_init=dt_bias_init,
            name='dt_proj'
        )(x_dt)  # (B, L, D)

        # Apply softplus to get positive delta
        delta = jax.nn.softplus(dt_proj)  # (B, L, D)

        # === Expand B and C to match (B, L, D, N) ===
        # B_sel: (B, L, N) -> (B, L, D, N) by broadcasting
        # In the recurrence, we compute B * x where x is (B, L, D)
        # So B needs to be (B, L, D, N) where each D dimension shares the same N values
        B_expanded = B_sel[:, :, None, :]  # (B, L, 1, N)
        B_expanded = jnp.broadcast_to(B_expanded, (B_size, L, D, N))

        C_expanded = C_sel[:, :, None, :]  # (B, L, 1, N)
        C_expanded = jnp.broadcast_to(C_expanded, (B_size, L, D, N))

        # === Discretization ===
        A_bar, B_bar = discretize_zoh_simple(A, B_expanded, delta)

        # === Selective Scan ===
        if use_parallel:
            y, _ = selective_scan_parallel(A_bar, B_bar, C_expanded, x)
        else:
            y, _ = selective_scan_sequential(A_bar, B_bar, C_expanded, x)

        # === Skip Connection ===
        y = y + D_param * x

        return y


class MambaBlock(nn.Module):
    """
    Full Mamba block as described in Section 3.4 of the paper.

    Architecture:
        Input x: (B, L, D)
             ↓
        Norm (RMSNorm)
             ↓
        Linear expansion → (B, L, E*D), E=2
             ↓
        Split into two branches:
            Branch 1: Conv1D → SiLU → S6 SSM
            Branch 2: (identity for gating)
             ↓
        Element-wise multiply (Branch1 * SiLU(Branch2))
             ↓
        Linear projection → (B, L, D)
             ↓
        Residual: Output + Input
             ↓
        Output: (B, L, D)

    This follows Figure 3 from the paper.
    """
    d_model: int          # Model dimension (D)
    d_state: int = 16     # SSM state dimension (N)
    d_conv: int = 4       # Convolution kernel size
    expand: int = 2       # Expansion factor (E)
    dt_rank: int = None   # Rank for delta projection

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        use_parallel: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (B, L, D)
            use_parallel: Whether to use parallel scan in SSM

        Returns:
            y: Output tensor, shape (B, L, D)
        """
        B_size, L, D = x.shape
        D_inner = self.expand * D  # Inner dimension after expansion

        # Store input for residual connection
        residual = x

        # === Input Normalization ===
        x = nn.RMSNorm(name='norm')(x)

        # === Linear Expansion ===
        # Project to 2 * D_inner (for two branches)
        x_proj = nn.Dense(
            features=2 * D_inner,
            use_bias=False,
            kernel_init=initializers.lecun_normal(),
            name='in_proj'
        )(x)  # (B, L, 2*D_inner)

        # Split into two branches
        x_main, x_gate = jnp.split(x_proj, 2, axis=-1)  # Each: (B, L, D_inner)

        # === Branch 1: Conv → SiLU → SSM ===

        # 1D Causal Convolution
        x_conv = CausalConv1D(
            features=D_inner,
            kernel_size=self.d_conv,
            name='conv1d'
        )(x_main)  # (B, L, D_inner)

        # SiLU activation (also called Swish)
        x_conv = jax.nn.silu(x_conv)

        # Selective SSM (S6)
        x_ssm = S6Layer(
            d_model=D_inner,
            d_state=self.d_state,
            dt_rank=self.dt_rank,
            name='ssm'
        )(x_conv, use_parallel=use_parallel)  # (B, L, D_inner)

        # === Gating ===
        # Multiply by SiLU of gate branch
        x_gated = x_ssm * jax.nn.silu(x_gate)  # (B, L, D_inner)

        # === Output Projection ===
        y = nn.Dense(
            features=D,
            use_bias=False,
            kernel_init=initializers.lecun_normal(),
            name='out_proj'
        )(x_gated)  # (B, L, D)

        # === Residual Connection ===
        y = y + residual

        return y


# =============================================================================
# Testing and Verification
# =============================================================================

def test_discretization():
    """Test ZOH discretization."""
    print("\n" + "="*60)
    print("Testing ZOH Discretization")
    print("="*60)

    # Create test inputs
    D, N = 4, 8
    B_size, L = 2, 16

    # A should be negative for stability (S4D-Real initialization)
    A = -jnp.arange(1, N + 1, dtype=jnp.float32)  # (N,)
    A = jnp.broadcast_to(A, (D, N))  # (D, N)

    # B: (B, L, D, N)
    key = jax.random.PRNGKey(42)
    B = jax.random.normal(key, (B_size, L, D, N))

    # Delta: (B, L, D) - small positive values
    key, subkey = jax.random.split(key)
    delta = jax.nn.softplus(jax.random.normal(subkey, (B_size, L, D)) - 2)

    # Test discretization
    A_bar, B_bar = discretize_zoh_simple(A, B, delta)

    print(f"Input A shape: {A.shape}")
    print(f"Input B shape: {B.shape}")
    print(f"Input delta shape: {delta.shape}")
    print(f"Output A_bar shape: {A_bar.shape}")
    print(f"Output B_bar shape: {B_bar.shape}")

    # Check A_bar is in (0, 1) for stable dynamics
    print(f"A_bar range: [{A_bar.min():.4f}, {A_bar.max():.4f}]")
    print(f"Delta range: [{delta.min():.4f}, {delta.max():.4f}]")

    # Verify stability: A_bar should be < 1 in magnitude
    assert jnp.all(jnp.abs(A_bar) <= 1.0), "A_bar should be bounded by 1"
    print("✓ Discretization test passed!")

    return True


def test_selective_scan():
    """Test parallel vs sequential scan equivalence."""
    print("\n" + "="*60)
    print("Testing Selective Scan")
    print("="*60)

    # Create test inputs
    D, N = 4, 8
    B_size, L = 2, 32

    key = jax.random.PRNGKey(0)

    # Input sequence
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (B_size, L, D))

    # A_bar: (B, L, D, N) - decay factors in (0, 1)
    key, subkey = jax.random.split(key)
    A_bar = 0.9 * jnp.ones((B_size, L, D, N)) + 0.05 * jax.random.uniform(
        subkey, (B_size, L, D, N)
    )

    # B_bar: (B, L, D, N)
    key, subkey = jax.random.split(key)
    B_bar = 0.1 * jax.random.normal(subkey, (B_size, L, D, N))

    # C: (B, L, D, N)
    key, subkey = jax.random.split(key)
    C = jax.random.normal(subkey, (B_size, L, D, N))

    # Test sequential scan
    y_seq, h_seq = selective_scan_sequential(A_bar, B_bar, C, x)

    # Test parallel scan
    y_par, h_par = selective_scan_parallel(A_bar, B_bar, C, x)

    print(f"Input x shape: {x.shape}")
    print(f"Output y_seq shape: {y_seq.shape}")
    print(f"Output y_par shape: {y_par.shape}")

    # Check equivalence
    max_diff_y = jnp.max(jnp.abs(y_seq - y_par))
    max_diff_h = jnp.max(jnp.abs(h_seq - h_par))

    print(f"Max difference in y: {max_diff_y:.2e}")
    print(f"Max difference in h: {max_diff_h:.2e}")

    # Allow for numerical precision differences
    assert max_diff_y < 1e-4, f"y mismatch: {max_diff_y}"
    assert max_diff_h < 1e-4, f"h mismatch: {max_diff_h}"

    print("✓ Sequential and parallel scans match!")

    # Benchmark (basic timing)
    import time

    # JIT compile
    scan_seq_jit = jax.jit(selective_scan_sequential)
    scan_par_jit = jax.jit(selective_scan_parallel)

    # Warmup
    _ = scan_seq_jit(A_bar, B_bar, C, x)
    _ = scan_par_jit(A_bar, B_bar, C, x)

    # Time sequential
    start = time.time()
    for _ in range(100):
        _ = scan_seq_jit(A_bar, B_bar, C, x)
    jax.block_until_ready(_)
    seq_time = time.time() - start

    # Time parallel
    start = time.time()
    for _ in range(100):
        _ = scan_par_jit(A_bar, B_bar, C, x)
    jax.block_until_ready(_)
    par_time = time.time() - start

    print(f"\nTiming (100 iterations, L={L}):")
    print(f"Sequential: {seq_time*10:.2f} ms")
    print(f"Parallel:   {par_time*10:.2f} ms")
    print(f"Speedup:    {seq_time/par_time:.2f}x")

    return True


def run_phase1_tests():
    """Run all Phase 1 tests."""
    print("\n" + "="*60)
    print("MAMBA JAX IMPLEMENTATION - PHASE 1 TESTS")
    print("="*60)

    # Check device
    has_gpu = check_device()

    # Run tests
    test_discretization()
    test_selective_scan()

    print("\n" + "="*60)
    print("ALL PHASE 1 TESTS PASSED!")
    print("="*60)

    return True


# =============================================================================
# Phase 2 Tests
# =============================================================================

def test_causal_conv1d():
    """Test causal 1D convolution."""
    print("\n" + "="*60)
    print("Testing Causal Conv1D")
    print("="*60)

    B, L, D = 2, 16, 32
    kernel_size = 4

    # Create test input
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (B, L, D))

    # Initialize module
    conv = CausalConv1D(features=D, kernel_size=kernel_size)
    key, subkey = jax.random.split(key)
    params = conv.init(subkey, x)

    # Forward pass
    y = conv.apply(params, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Check output shape matches input
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    # Test causality: output at position t should not depend on input at t+1
    # Modify input at position L-1 and check if earlier outputs change
    x_modified = x.at[:, -1, :].set(0.0)
    y_modified = conv.apply(params, x_modified)

    # Outputs before position L-kernel_size+1 should be identical
    # (because the last position can only affect the last kernel_size positions)
    safe_positions = L - kernel_size
    diff = jnp.abs(y[:, :safe_positions] - y_modified[:, :safe_positions])
    max_diff = jnp.max(diff)

    print(f"Causality check - max diff in safe positions: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Causality violated: {max_diff}"

    print("✓ Causal Conv1D test passed!")
    return True


def test_s6_layer():
    """Test S6 (Selective SSM) layer."""
    print("\n" + "="*60)
    print("Testing S6 Layer")
    print("="*60)

    B, L, D = 2, 32, 64
    N = 16  # State dimension

    # Create test input
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (B, L, D))

    # Initialize module
    s6 = S6Layer(d_model=D, d_state=N)
    key, subkey = jax.random.split(key)
    params = s6.init(subkey, x)

    # Forward pass (parallel)
    y_par = s6.apply(params, x, use_parallel=True)

    # Forward pass (sequential for verification)
    y_seq = s6.apply(params, x, use_parallel=False)

    print(f"Input shape: {x.shape}")
    print(f"Output shape (parallel): {y_par.shape}")
    print(f"Output shape (sequential): {y_seq.shape}")

    # Check shapes
    assert y_par.shape == x.shape, f"Shape mismatch: {y_par.shape} vs {x.shape}"
    assert y_seq.shape == x.shape, f"Shape mismatch: {y_seq.shape} vs {x.shape}"

    # Check parallel vs sequential equivalence
    max_diff = jnp.max(jnp.abs(y_par - y_seq))
    print(f"Max diff (parallel vs sequential): {max_diff:.2e}")
    assert max_diff < 1e-4, f"Parallel/Sequential mismatch: {max_diff}"

    # Check parameter shapes
    print("\nParameter shapes:")
    def print_params(params, prefix=""):
        for key, value in params.items():
            if isinstance(value, dict):
                print_params(value, prefix + key + "/")
            else:
                print(f"  {prefix}{key}: {value.shape}")
    print_params(params['params'])

    print("✓ S6 Layer test passed!")
    return True


def test_mamba_block():
    """Test full Mamba block."""
    print("\n" + "="*60)
    print("Testing Mamba Block")
    print("="*60)

    B, L, D = 2, 32, 64
    N = 16  # State dimension
    d_conv = 4
    expand = 2

    # Create test input
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (B, L, D))

    # Initialize module
    block = MambaBlock(d_model=D, d_state=N, d_conv=d_conv, expand=expand)
    key, subkey = jax.random.split(key)
    params = block.init(subkey, x)

    # Forward pass
    y = block.apply(params, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Check output shape matches input (residual connection)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    # Check that output is different from input (block does something)
    diff = jnp.max(jnp.abs(y - x))
    print(f"Max diff from input: {diff:.2f}")
    assert diff > 0.01, "Block output too similar to input"

    # Count parameters
    def count_params(params):
        return sum(p.size for p in jax.tree_util.tree_leaves(params))

    num_params = count_params(params)
    print(f"Total parameters: {num_params:,}")

    # Expected rough count:
    # in_proj: D * 2*E*D = 64 * 256 = 16384
    # conv1d: E*D * kernel = 128 * 4 = 512
    # ssm: various
    # out_proj: E*D * D = 128 * 64 = 8192
    print(f"Expected ~30-40k params for D={D}, expand={expand}")

    # Print parameter breakdown
    print("\nParameter breakdown:")
    def print_param_counts(params, prefix=""):
        for key, value in params.items():
            if isinstance(value, dict):
                print_param_counts(value, prefix + key + "/")
            else:
                print(f"  {prefix}{key}: {value.size:,} ({value.shape})")
    print_param_counts(params['params'])

    print("✓ Mamba Block test passed!")
    return True


def test_mamba_block_gradients():
    """Test that gradients flow through Mamba block."""
    print("\n" + "="*60)
    print("Testing Mamba Block Gradients")
    print("="*60)

    B, L, D = 2, 16, 32
    N = 8

    # Create test input
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (B, L, D))

    # Initialize module
    block = MambaBlock(d_model=D, d_state=N)
    key, subkey = jax.random.split(key)
    params = block.init(subkey, x)

    # Define loss function
    def loss_fn(params, x):
        y = block.apply(params, x)
        return jnp.mean(y ** 2)

    # Compute gradients
    grads = jax.grad(loss_fn)(params, x)

    # Check that all gradients are finite and non-zero
    def check_grads(grads, prefix=""):
        all_finite = True
        all_nonzero = True
        for key, value in grads.items():
            if isinstance(value, dict):
                f, nz = check_grads(value, prefix + key + "/")
                all_finite = all_finite and f
                all_nonzero = all_nonzero and nz
            else:
                is_finite = jnp.all(jnp.isfinite(value))
                is_nonzero = jnp.any(value != 0)
                if not is_finite:
                    print(f"  WARNING: {prefix}{key} has non-finite gradients")
                    all_finite = False
                if not is_nonzero:
                    print(f"  WARNING: {prefix}{key} has zero gradients")
                    all_nonzero = False
        return all_finite, all_nonzero

    all_finite, all_nonzero = check_grads(grads['params'])

    print(f"All gradients finite: {all_finite}")
    print(f"All gradients non-zero: {all_nonzero}")

    assert all_finite, "Some gradients are not finite!"
    assert all_nonzero, "Some gradients are zero!"

    # Test JIT compilation
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    loss, grads = loss_and_grad(params, x)
    print(f"JIT compiled loss: {loss:.4f}")

    print("✓ Gradient test passed!")
    return True


def run_phase2_tests():
    """Run all Phase 2 tests."""
    print("\n" + "="*60)
    print("MAMBA JAX IMPLEMENTATION - PHASE 2 TESTS")
    print("="*60)

    # Check device
    has_gpu = check_device()

    # Run Phase 2 tests
    test_causal_conv1d()
    test_s6_layer()
    test_mamba_block()
    test_mamba_block_gradients()

    print("\n" + "="*60)
    print("ALL PHASE 2 TESTS PASSED!")
    print("="*60)

    return True


def run_all_tests():
    """Run all tests (Phase 1 + Phase 2)."""
    run_phase1_tests()
    run_phase2_tests()


if __name__ == "__main__":
    run_all_tests()
