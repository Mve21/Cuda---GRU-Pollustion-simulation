// Region profiles live in constant memory: they are read by every thread of
// the bonus kernels but never written, which is exactly what __constant__ is
// optimised for. The host fills this array once via cuMemcpyHtoD.
__constant__ RegionProfile d_region_profiles[MAX_TYPES];
