#pragma once
#include "stratum/ecdf_miner.h"   // ecdf_state_t, ecdf_result_t — pure math, no cycle
#include <esp_err.h>
#include <map>
#include <vector>

// Forward declarations — no #include of miner.h or stratum.h needed
class  AsicMinerClass;
struct pool_job_data_t;



// ── Per-difficulty ECDF state ─────────────────────────────────────────────────
struct ecdf_tier_t {
    ecdf_state_t state  = {};
    uint32_t     diff   = 0; 
    uint32_t     anchor = 0; 
};

struct ecdf_multi_t {
    std::map<uint32_t, ecdf_tier_t> tiers;

    ecdf_tier_t& get(uint32_t diff) {
        if (tiers.find(diff) == tiers.end()) {
            ecdf_tier_t t;
            t.diff = diff;
            ecdf_reset(t.state);
            tiers[diff] = t;
        }
        return tiers[diff];
    }

    void reset_all() { tiers.clear(); }
};

struct ecdf_share_t {
    uint32_t nonce;
    uint32_t version;
    uint8_t  asic_id;
    uint8_t  job_id;
};

uint32_t ecdf_scan_window_asic(ecdf_multi_t              &multi,
                               AsicMinerClass            &asic,
                               pool_job_data_t           *pool_job,
                               uint32_t                   nonce_period_ms,
                               std::vector<ecdf_share_t> &out_shares);
