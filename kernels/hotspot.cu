// Per-region statistics and hotspot detection (uniform alpha) -- basic stage.
//
// Launched with one thread block per region; the block dimensions equal the
// region size (region_cols x region_rows). Each block:
//   1. reduces the region's danger counts into a mean,
//   2. reduces the squared deviations into a standard deviation,
//   3. flags every cell whose danger count exceeds mean + alpha * stddev.
//
// Shared memory holds two int-sized buffers of `block_size` elements:
//   original_data -- the untouched danger counts (needed in step 2)
//   temp_results  -- scratch space reused by both parallel reductions.
//
// NOTE: the tree reduction assumes block_size (region_rows * region_cols) is a
// power of two; the host validates this before launching.
extern "C"
__global__ void region_hotspot_stats_kernel(const int *danger_count,
                                            float *region_means,
                                            float *region_deviation,
                                            int8_t *is_hotspot,
                                            int total_cols, int region_rows,
                                            int region_cols, float alpha) {
    int region_row = blockIdx.y;
    int region_col = blockIdx.x;
    int region_idx = region_row * gridDim.x + region_col;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int tid = local_row * blockDim.x + local_col;

    int global_row = region_row * region_rows + local_row;
    int global_col = region_col * region_cols + local_col;
    int global_idx = global_row * total_cols + global_col;

    int block_size = region_rows * region_cols;

    extern __shared__ int shared_mem[];
    int *original_data = shared_mem;
    int *temp_results  = &shared_mem[block_size];

    original_data[tid] = danger_count[global_idx];
    temp_results[tid]  = original_data[tid];
    __syncthreads();

    // Reduction 1: sum of danger counts -> mean.
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) temp_results[tid] += temp_results[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        region_means[region_idx] = (float)temp_results[0] / (float)block_size;
    }
    __syncthreads();

    // Reduction 2: sum of squared deviations -> standard deviation.
    // The scratch buffer is reinterpreted as float for the variance pass.
    float *temp_float = (float *)temp_results;
    float diff = (float)original_data[tid] - region_means[region_idx];
    temp_float[tid] = diff * diff;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) temp_float[tid] += temp_float[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        region_deviation[region_idx] = sqrtf(temp_float[0] / (float)block_size);
    }
    __syncthreads();

    // Flag the cell as a hotspot if it stands out within its region.
    is_hotspot[global_idx] = (int8_t)(danger_count[global_idx] >
        region_means[region_idx] + alpha * region_deviation[region_idx]);
}
