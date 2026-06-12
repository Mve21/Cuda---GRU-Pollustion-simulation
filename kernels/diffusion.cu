// One explicit time step of the diffusion process.
//
// Every cell relaxes towards the average of itself and its four direct
// neighbours (a 5-point stencil). Cells outside the grid are treated as 0.
// The result is scaled by (1 - decay) so the substance slowly dissipates.
//
// Read from `conc`, write to `conc_out` (double buffering) to avoid races.
extern "C"
__global__ void diffusion_step_kernel(const float *conc, float *conc_out,
                                      float decay, int H, int W) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        int idx = row * W + col;

        float center = conc[idx];
        float up    = (row > 0)     ? conc[(row - 1) * W + col] : 0.0f;
        float down  = (row < H - 1) ? conc[(row + 1) * W + col] : 0.0f;
        float left  = (col > 0)     ? conc[row * W + (col - 1)] : 0.0f;
        float right = (col < W - 1) ? conc[row * W + (col + 1)] : 0.0f;

        float avg = (center + up + down + left + right) / 5.0f;
        conc_out[idx] = avg * (1.0f - decay);
    }
}
