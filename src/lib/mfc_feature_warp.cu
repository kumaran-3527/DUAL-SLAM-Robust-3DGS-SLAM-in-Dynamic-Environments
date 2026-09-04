#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel that fuses Bilinear Interpolation, Cosine Similarity, and SCE Thresholding
__global__ void sce_feature_warp_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> feats_ii,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> feats_jj,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> grid,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> geom_valid,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> sce_f)
{
    // Global pixel coordinates
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int e = blockIdx.z * blockDim.z + threadIdx.z;

    int E = sce_f.size(0);
    int H = sce_f.size(1);
    int W = sce_f.size(2);
    int C = feats_ii.size(1); // 384 channels for DINO

    // Bounds check
    if (e >= E || h >= H || w >= W) return;

    // Geometric validity gate (if point is invalid geometrically, SCE = 0)
    if (!geom_valid[e][h][w]) {
        sce_f[e][h][w] = 0.0f;
        return;
    }

    // Read normalized grid coordinates [-1, 1]
    float nx = grid[e][h][w][0];
    float ny = grid[e][h][w][1];

    // Denormalize to pixel coordinates [0, W-1] (Matches align_corners=True)
    float px = (nx + 1.0f) * (W - 1.0f) * 0.5f;
    float py = (ny + 1.0f) * (H - 1.0f) * 0.5f;

    // Bilinear interpolation corners
    int x0 = (int)floorf(px);
    int y0 = (int)floorf(py);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Bilinear interpolation weights
    float dx = px - x0;
    float dy = py - y0;
    float wx0y0 = (1.0f - dx) * (1.0f - dy);
    float wx1y0 = dx * (1.0f - dy);
    float wx0y1 = (1.0f - dx) * dy;
    float wx1y1 = dx * dy;

    // Boundary checks (padding_mode='zeros')
    bool p00_valid = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H);
    bool p10_valid = (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H);
    bool p01_valid = (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H);
    bool p11_valid = (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H);

    float dot_product = 0.0f;

    // Loop over the 384 channels (Compute dot product instantly on the fly)
    // This entirely avoids creating the massive [E, 384, H, W] warped intermediate tensor in VRAM!
    for (int c = 0; c < C; ++c) {
        float val_jj = 0.0f;
        if (p00_valid) val_jj += wx0y0 * feats_jj[e][c][y0][x0];
        if (p10_valid) val_jj += wx1y0 * feats_jj[e][c][y0][x1];
        if (p01_valid) val_jj += wx0y1 * feats_jj[e][c][y1][x0];
        if (p11_valid) val_jj += wx1y1 * feats_jj[e][c][y1][x1];

        float val_ii = feats_ii[e][c][h][w];
        
        // Accumulate dot product (features must be L2-normalized before passing to CUDA)
        dot_product += val_ii * val_jj;
    }

    // Post-processing constraints matching Python exactly
    if (dot_product < 0.0f) dot_product = 0.0f;
    if (dot_product > 1.0f) dot_product = 1.0f;
    
    // Hard threshold (zero out low-confidence matches)
    if (dot_product < 0.50f) dot_product = 0.0f;

    // Calculate final Semantic Consistency Error (SCE)
    sce_f[e][h][w] = 1.0f - dot_product;
}

// C++ Binding function
torch::Tensor sce_feature_warp_cuda(
    torch::Tensor feats_ii,
    torch::Tensor feats_jj,
    torch::Tensor grid,
    torch::Tensor geom_valid)
{
    // Ensure tensors are contiguous in memory before accessing pointers
    feats_ii = feats_ii.contiguous();
    feats_jj = feats_jj.contiguous();
    grid = grid.contiguous();
    geom_valid = geom_valid.contiguous();

    int E = grid.size(0);
    int H = grid.size(1);
    int W = grid.size(2);

    // Initialize the final output tensor
    auto sce_f = torch::zeros({E, H, W}, grid.options());

    // Threads and Block dimensions
    dim3 blocks((W + 15) / 16, (H + 15) / 16, E);
    dim3 threads(16, 16, 1);

    // Launch Kernel
    sce_feature_warp_kernel<<<blocks, threads>>>(
        feats_ii.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        feats_jj.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        grid.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        geom_valid.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
        sce_f.packed_accessor32<float,3,torch::RestrictPtrTraits>()
    );

    return sce_f;
}

// Forward declaration for the function defined in mfc_warping.cu
std::vector<torch::Tensor> sce_warp_cuda(
    torch::Tensor depth,
    torch::Tensor Gij,
    torch::Tensor intr_src,
    torch::Tensor intr_tgt);

__global__ void sce_feature_warp_full_res_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> feats_ii,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> feats_jj,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> x1_b, // [E, H_full, W_full, 3]
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_mask, // [E, H_full, W_full]
    torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> matching_mask) // [E, H_full, W_full]
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int e = blockIdx.z * blockDim.z + threadIdx.z;

    int H_full = x1_b.size(1);
    int W_full = x1_b.size(2);
    int C = feats_ii.size(1);
    int H_feat = feats_ii.size(2);
    int W_feat = feats_ii.size(3);

    if (h >= H_full || w >= W_full || e >= x1_b.size(0)) return;

    if (!valid_mask[e][h][w]) {
        matching_mask[e][h][w] = false;
        return;
    }

    // PyTorch F.interpolate(align_corners=False) exact scaling math
    float scale_w = (float)W_full / (float)W_feat;
    float scale_h = (float)H_full / (float)H_feat;

    // Coordinates in the TARGET image (j), scaled down to feature resolution
    float px = (x1_b[e][h][w][0] + 0.5f) / scale_w - 0.5f;
    float py = (x1_b[e][h][w][1] + 0.5f) / scale_h - 0.5f;

    // Coordinates in the SOURCE image (i), scaled down
    float ix = ((float)w + 0.5f) / scale_w - 0.5f;
    float iy = ((float)h + 0.5f) / scale_h - 0.5f;

    // --- Interpolation for feats_jj (at px, py) ---
    int x0_j = (int)floorf(px);
    int y0_j = (int)floorf(py);
    int x1_j = x0_j + 1;
    int y1_j = y0_j + 1;

    float dx_j = px - (float)x0_j;
    float dy_j = py - (float)y0_j;
    float wx0y0_j = (1.0f - dx_j) * (1.0f - dy_j);
    float wx1y0_j = dx_j * (1.0f - dy_j);
    float wx0y1_j = (1.0f - dx_j) * dy_j;
    float wx1y1_j = dx_j * dy_j;

    bool p00_j_valid = (x0_j >= 0 && x0_j < W_feat && y0_j >= 0 && y0_j < H_feat);
    bool p10_j_valid = (x1_j >= 0 && x1_j < W_feat && y0_j >= 0 && y0_j < H_feat);
    bool p01_j_valid = (x0_j >= 0 && x0_j < W_feat && y1_j >= 0 && y1_j < H_feat);
    bool p11_j_valid = (x1_j >= 0 && x1_j < W_feat && y1_j >= 0 && y1_j < H_feat);

    // --- Interpolation for feats_ii (at ix, iy) ---
    int x0_i = (int)floorf(ix);
    int y0_i = (int)floorf(iy);
    int x1_i = x0_i + 1;
    int y1_i = y0_i + 1;

    float dx_i = ix - (float)x0_i;
    float dy_i = iy - (float)y0_i;
    float wx0y0_i = (1.0f - dx_i) * (1.0f - dy_i);
    float wx1y0_i = dx_i * (1.0f - dy_i);
    float wx0y1_i = (1.0f - dx_i) * dy_i;
    float wx1y1_i = dx_i * dy_i;

    bool p00_i_valid = (x0_i >= 0 && x0_i < W_feat && y0_i >= 0 && y0_i < H_feat);
    bool p10_i_valid = (x1_i >= 0 && x1_i < W_feat && y0_i >= 0 && y0_i < H_feat);
    bool p01_i_valid = (x0_i >= 0 && x0_i < W_feat && y1_i >= 0 && y1_i < H_feat);
    bool p11_i_valid = (x1_i >= 0 && x1_i < W_feat && y1_i >= 0 && y1_i < H_feat);

    float dot_product = 0.0f;

    for (int c = 0; c < C; ++c) {
        float val_jj = 0.0f;
        if (p00_j_valid) val_jj += wx0y0_j * feats_jj[e][c][y0_j][x0_j];
        if (p10_j_valid) val_jj += wx1y0_j * feats_jj[e][c][y0_j][x1_j];
        if (p01_j_valid) val_jj += wx0y1_j * feats_jj[e][c][y1_j][x0_j];
        if (p11_j_valid) val_jj += wx1y1_j * feats_jj[e][c][y1_j][x1_j];

        float val_ii = 0.0f;
        if (p00_i_valid) val_ii += wx0y0_i * feats_ii[e][c][y0_i][x0_i];
        if (p10_i_valid) val_ii += wx1y0_i * feats_ii[e][c][y0_i][x1_i];
        if (p01_i_valid) val_ii += wx0y1_i * feats_ii[e][c][y1_i][x0_i];
        if (p11_i_valid) val_ii += wx1y1_i * feats_ii[e][c][y1_i][x1_i];
        
        dot_product += val_ii * val_jj;
    }

    matching_mask[e][h][w] = (dot_product > 0.90f);
}

torch::Tensor sce_feature_warp_full_res_cuda(
    torch::Tensor feats_ii,
    torch::Tensor feats_jj,
    torch::Tensor x1_b,
    torch::Tensor valid_mask)
{
    feats_ii = feats_ii.contiguous();
    feats_jj = feats_jj.contiguous();
    x1_b = x1_b.contiguous();
    valid_mask = valid_mask.contiguous();

    int E = x1_b.size(0);
    int H = x1_b.size(1);
    int W = x1_b.size(2);

    auto matching_mask = torch::zeros({E, H, W}, valid_mask.options());

    dim3 blocks((W + 15) / 16, (H + 15) / 16, E);
    dim3 threads(16, 16, 1);

    sce_feature_warp_full_res_kernel<<<blocks, threads>>>(
        feats_ii.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        feats_jj.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        x1_b.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        valid_mask.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
        matching_mask.packed_accessor32<bool,3,torch::RestrictPtrTraits>()
    );

    return matching_mask;
}

// Bind to Python via PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sce_feature_warp", &sce_feature_warp_cuda, "Fused SCE Feature Warping and Dot Product (CUDA)");
    m.def("sce_feature_warp_full_res", &sce_feature_warp_full_res_cuda, "Fused Full-Resolution SCE Mask (CUDA)");
    m.def("sce_warp", &sce_warp_cuda, "SCE Coordinate Warping (CUDA)");
}
