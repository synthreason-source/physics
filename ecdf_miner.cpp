#include "ecdf_miner.h"

void ecdf_reset(ecdf_state_t &s) {
    s.mean_gap = 0.0f;
    s.m2 = 0.0f;
    s.cv = 0.0f;
    s.n_gaps = 0;
    s.anchor = 0;
    s.fitted = false;
}

bool ecdf_add_gap(ecdf_state_t &s, uint32_t gap) {
    s.n_gaps++;
    float delta = (float)gap - s.mean_gap;
    s.mean_gap += delta / (float)s.n_gaps;
    float delta2 = (float)gap - s.mean_gap;
    s.m2 += delta * delta2;

    if (s.n_gaps >= 2) {
        float variance = s.m2 / (float)(s.n_gaps - 1);
        float stddev = (variance > 0.0f) ? sqrtf(variance) : 0.0f;
        s.cv = (s.mean_gap > 0.0f) ? (stddev / s.mean_gap) : 0.0f;
    }
    if (s.n_gaps >= ECDF_MIN_GAPS) {
        s.fitted = true;
    }
    return s.fitted;
}

uint32_t ecdf_window(const ecdf_state_t &s, float p) {
    if (!s.fitted || s.mean_gap <= 0.0f) return 1u;
    if (p <= 0.0f) p = 0.01f;
    if (p >= 1.0f) p = 0.999f;
    float w = -s.mean_gap * logf(1.0f - p);
    return (w < 1.0f) ? 1u : (uint32_t)(w + 0.5f);
}

float ecdf_conf(const ecdf_state_t &s, uint32_t budget) {
    if (!s.fitted || s.mean_gap <= 0.0f) return 0.0f;
    return 1.0f - expf(-(float)budget / s.mean_gap);
}

float ecdf_marginal(const ecdf_state_t &s, uint32_t budget) {
    if (!s.fitted || s.mean_gap <= 0.0f) return 0.0f;
    return expf(-(float)budget / s.mean_gap) / s.mean_gap;
}

uint32_t ecdf_calibrate(ecdf_state_t &s, uint32_t budget, bool (*is_valid_fn)(uint32_t nonce)) {
    uint32_t hits = 0;
    uint32_t prev = 0;
    bool have_prev = false;

    for (uint32_t n = 0; n < budget; n++) {
        if (is_valid_fn(n)) {
            hits++;
            if (have_prev) {
                ecdf_add_gap(s, n - prev);
            }
            prev = n;
            have_prev = true;
            s.anchor = n;
            if (s.fitted) break;
        }
    }
    return hits;
}

ecdf_result_t ecdf_scan_window(ecdf_state_t &s, float conf, bool (*is_valid_fn)(uint32_t nonce)) {
    ecdf_result_t r = {false, 0, 0, conf};
    if (conf <= 0.0f) conf = ECDF_DEFAULT_CONF;
    r.conf_actual = conf;

    uint32_t w = ecdf_window(s, conf);
    uint32_t start = s.anchor + 1;
    uint32_t end = s.anchor + w;
    if (end < start) end = 0xFFFFFFFFu; // guard wrap

    for (uint32_t n = start; n <= end; n++) {
        if (is_valid_fn(n)) {
            r.hit = true;
            r.nonce = n;
            r.hashes = n - s.anchor;
            s.anchor = n;
            ecdf_add_gap(s, r.hashes);
            return r;
        }
    }
    r.hashes = w;
    s.anchor += w;
    return r;
}

uint32_t ecdf_mine(ecdf_state_t &s, float conf, uint32_t target_hits, uint32_t max_hashes,
                   bool (*is_valid_fn)(uint32_t nonce), void (*on_hit_fn)(uint32_t nonce)) {
    if (conf <= 0.0f) conf = ECDF_DEFAULT_CONF;
    uint32_t total = 0;
    uint32_t hits = 0;

    while (hits < target_hits && total < max_hashes) {
        ecdf_result_t r = ecdf_scan_window(s, conf, is_valid_fn);
        total += r.hashes;
        if (r.hit) {
            hits++;
            if (on_hit_fn) on_hit_fn(r.nonce);
        }
    }
    return total;
}
