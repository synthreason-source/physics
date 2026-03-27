#pragma once
#include <stdint.h>
#include <math.h>

// ── Compile-time knobs ────────────────────────────────────────────────────────
#ifndef ECDF_MIN_GAPS
# define ECDF_MIN_GAPS 8
#endif
#ifndef ECDF_DEFAULT_CONF
# define ECDF_DEFAULT_CONF 0.90f
#endif
#ifndef ECDF_CALIB_BUDGET
# define ECDF_CALIB_BUDGET 5000000000000u
#endif

struct ecdf_state_t {
    float    mean_gap;
    float    m2;
    float    cv;
    uint32_t n_gaps;
    uint32_t anchor;
    bool     fitted;
};

struct ecdf_result_t {
    bool     hit;
    uint32_t nonce;
    uint32_t hashes;
    float    conf_actual;
};

void ecdf_reset(ecdf_state_t &s);
bool ecdf_add_gap(ecdf_state_t &s, uint32_t gap);
uint32_t ecdf_window(const ecdf_state_t &s, float p);
float ecdf_conf(const ecdf_state_t &s, uint32_t budget);
float ecdf_marginal(const ecdf_state_t &s, uint32_t budget);

uint32_t ecdf_calibrate(ecdf_state_t &s, uint32_t budget, bool (*is_valid_fn)(uint32_t));
ecdf_result_t ecdf_scan_window(ecdf_state_t &s, float conf, bool (*is_valid_fn)(uint32_t));
uint32_t ecdf_mine(ecdf_state_t &s, float conf, uint32_t target_hits, uint32_t max_hashes, 
                   bool (*is_valid_fn)(uint32_t), void (*on_hit_fn)(uint32_t));
