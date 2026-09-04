#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward rotation by quaternion (assuming qx, qy, qz, qw)
__device__ void actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0f * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0f * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0f * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

// Rigid transformation
__device__ void actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[0] += t[0];
  Y[1] += t[1];
  Y[2] += t[2];
}

__global__ void sce_warp_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> depth, // [E, H, W]
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Gij,   // [E, 7] (tx, ty, tz, qx, qy, qz, qw)
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intr_src, // [E, 4]
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intr_tgt, // [E, 4]
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> grid,        // [E, H, W, 2]
    torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> geom_valid)   // [E, H, W]
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int e = blockIdx.z * blockDim.z + threadIdx.z;

    int W = depth.size(2);
    int H = depth.size(1);
    int E = depth.size(0);

    if (e >= E || h >= H || w >= W) return;

    float d = depth[e][h][w];

    // Source intrinsics
    float fx0 = intr_src[e][0];
    float fy0 = intr_src[e][1];
    float cx0 = intr_src[e][2];
    float cy0 = intr_src[e][3];

    // Target intrinsics
    float fx1 = intr_tgt[e][0];
    float fy1 = intr_tgt[e][1];
    float cx1 = intr_tgt[e][2];
    float cy1 = intr_tgt[e][3];

    // 1. Lift to 3D in source camera frame (iproj)
    float X0[3];
    X0[0] = (w - cx0) / fx0 * d;
    X0[1] = (h - cy0) / fy0 * d;
    X0[2] = d;

    // 2. Rigid transform (actp)
    float t[3] = { Gij[e][0], Gij[e][1], Gij[e][2] };
    float q[4] = { Gij[e][3], Gij[e][4], Gij[e][5], Gij[e][6] }; // qx, qy, qz, qw
    float X1[3];
    actSE3(t, q, X0, X1);

    // 3. Project to target camera frame (proj)
    float u1 = (X1[0] / X1[2]) * fx1 + cx1;
    float v1 = (X1[1] / X1[2]) * fy1 + cy1;

    // 4. Bounds check and MIN_DEPTH check (0.25m threshold)
    bool valid = (X1[2] >= 0.25f) && 
                 (u1 >= 0.0f) && (u1 <= W - 1.0f) && 
                 (v1 >= 0.0f) && (v1 <= H - 1.0f);

    // 5. Compute Normalized Grid coords [-1, 1] for F.grid_sample
    grid[e][h][w][0] = 2.0f * u1 / (W - 1.0f) - 1.0f;
    grid[e][h][w][1] = 2.0f * v1 / (H - 1.0f) - 1.0f;
    geom_valid[e][h][w] = valid;
}

std::vector<torch::Tensor> sce_warp_cuda(
    torch::Tensor depth,
    torch::Tensor Gij,
    torch::Tensor intr_src,
    torch::Tensor intr_tgt) 
{
    int E = depth.size(0);
    int H = depth.size(1);
    int W = depth.size(2);

    auto grid = torch::zeros({E, H, W, 2}, depth.options());
    auto geom_valid = torch::zeros({E, H, W}, depth.options().dtype(torch::kBool));

    // Simple 16x16 2D threading over pixels, stacked by edges in the Z dimension
    dim3 blocks((W + 15) / 16, (H + 15) / 16, E);
    dim3 threads(16, 16, 1);

    sce_warp_kernel<<<blocks, threads>>>(
        depth.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        Gij.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        intr_src.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        intr_tgt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        grid.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        geom_valid.packed_accessor32<bool,3,torch::RestrictPtrTraits>()
    );

    return {grid, geom_valid};
}

