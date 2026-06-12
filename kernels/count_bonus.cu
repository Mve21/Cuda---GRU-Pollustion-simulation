// Per-cell danger counter (per-region threshold) -- bonus stage.
//
// Identical in spirit to count_danger_kernel, but the threshold is no longer
// global: each region has a type, and each type carries its own
// danger_threshold via the constant-memory region profiles.
extern "C"
__global__ void count_danger_kernel_bonus(const float *conc, int *danger_count,
                                          const int *region_type,
                                          int H, int W, int R, int C,
                                          int total_region_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        int idx = row * W + col;

        // Map the cell to its region and look up that region's profile.
        int region_row = row / R;
        int region_col = col / C;
        int region_id  = region_row * total_region_cols + region_col;

        int type = region_type[region_id];
        float threshold = d_region_profiles[type].danger_threshold;

        if (conc[idx] >= threshold) {
            danger_count[idx]++;
        }
    }
}
