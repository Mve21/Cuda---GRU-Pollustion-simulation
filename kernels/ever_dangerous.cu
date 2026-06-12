// Count how many cells were dangerous at least once during the simulation.
//
// A cell qualifies when its accumulated danger_count is greater than zero.
// The global total is built with a single atomicAdd per qualifying cell.
extern "C"
__global__ void count_ever_dangerous_kernel(const int *danger_count,
                                            int H, int W, int *ever_dangerous) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        int idx = row * W + col;
        if (danger_count[idx] > 0) {
            atomicAdd(ever_dangerous, 1);
        }
    }
}
