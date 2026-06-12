// Apply sources and absorbers after each diffusion step.
//
// cell_type encodes the role of every cell:
//   0 -> normal cell, left untouched
//   1 -> source,   its concentration is pinned to `source_value`
//   2 -> absorber, `absorb_value` is subtracted (clamped at 0)
extern "C"
__global__ void apply_sources_kernel(const int8_t *cell_type, float *conc,
                                     float source_value, float absorb_value,
                                     int H, int W) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        int idx = row * W + col;
        int8_t type = cell_type[idx];

        if (type == 1) {
            conc[idx] = source_value;
        } else if (type == 2) {
            float reduced = conc[idx] - absorb_value;
            conc[idx] = (reduced > 0.0f) ? reduced : 0.0f;
        }
    }
}
