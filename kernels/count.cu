// Per-cell danger counter (uniform threshold).
//
// For every cell whose concentration is at or above `danger_threshold`
// at the current time step, its danger counter is incremented by one.
// Over the whole simulation danger_count[idx] therefore holds the number
// of steps during which the cell was dangerous.
extern "C"
__global__ void count_danger_kernel(const float *conc, int *danger_count,
                                    float danger_threshold, int H, int W) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        int idx = row * W + col;
        danger_count[idx] += (conc[idx] >= danger_threshold);
    }
}
