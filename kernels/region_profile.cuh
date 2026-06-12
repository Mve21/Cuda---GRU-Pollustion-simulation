#ifndef REGION_PROFILE_CUH
#define REGION_PROFILE_CUH

// Maximum number of distinct region profiles supported by the bonus stage.
#define MAX_TYPES 4

// Per-region-type parameters used by the bonus (advanced) analysis:
//   alpha            -> sensitivity of the hotspot test (mean + alpha * stddev)
//   danger_threshold -> concentration level that counts as "dangerous"
typedef struct {
    float alpha;
    float danger_threshold;
} RegionProfile;

#endif
