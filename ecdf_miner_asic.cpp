#include "ecdf_miner_asic.h"
#include "mining/miner.h"       // AsicMinerClass — full definition
#include "stratum/stratum.h"    // pool_job_data_t — full definition
#include "utils/logger/logger.h"
#include <Arduino.h>            // millis()

uint32_t ecdf_scan_window_asic(ecdf_multi_t              &multi,
                               AsicMinerClass            &asic,
                               pool_job_data_t           *pool_job,
                               uint32_t                   nonce_period_ms,
                               std::vector<ecdf_share_t> &out_shares)
{
    out_shares.clear();

    uint32_t     diff = (uint32_t)asic.get_asic_diff();
    ecdf_tier_t &tier = multi.get(diff);
    ecdf_state_t &s   = tier.state;

    uint32_t w;
    if (s.fitted) {
        w = ecdf_window(s, ECDF_DEFAULT_CONF);
    } else {
        w = (diff > 0) ? diff : 65536u;
    }

    uint32_t start = s.anchor + 1;
    uint32_t end   = s.anchor + w;
    if (end < start) end = 0xFFFFFFFFu;

    pool_job->starting_nonce = start;
    if (!asic.mining(pool_job)) {
        LOG_E("ecdf_scan_window_asic: mining() failed");
        s.anchor += w;
        return w;
    }

    const uint32_t MAX_TIMEOUT_MS = 8000u;
    const uint32_t PER_LISTEN_MS  = 200u;

    uint64_t budget_ms = (uint64_t)w * nonce_period_ms;
    if (budget_ms > MAX_TIMEOUT_MS) budget_ms = MAX_TIMEOUT_MS;

    uint32_t deadline      = millis() + (uint32_t)budget_ms;
    uint32_t last_hit_nonce = 0;
    bool     any_hit       = false;

    while ((int32_t)(deadline - millis()) > 0) {
        uint32_t remaining = deadline - millis();
        uint32_t slice     = (remaining < PER_LISTEN_MS) ? remaining : PER_LISTEN_MS;

        miner_result result;
        esp_err_t err = asic.listen_asic_rsp(&result, slice);

        if (err == ESP_ERR_TIMEOUT) continue;
        if (err != ESP_OK) {
            LOG_W("ecdf_scan_window_asic: listen error %d", err);
            continue;
        }

        uint32_t n = result.asic.nonce;

        ecdf_share_t share;
        share.nonce   = n;
        share.version = result.asic.version;
        share.asic_id = result.asic_id;
        share.job_id  = result.asic.job_id;
        out_shares.push_back(share);

        if (n > s.anchor) {
            ecdf_add_gap(s, n - s.anchor);
        }

        if (!any_hit || n > last_hit_nonce) {
            last_hit_nonce = n;
            any_hit = true;
        }

        LOG_D("ecdf tier[%u]: nonce=0x%08x job=%d asic=%d fitted=%d mean_gap=%.0f",
              diff, n, share.job_id, share.asic_id, s.fitted, s.mean_gap);
    }

    if (any_hit) {
        s.anchor = last_hit_nonce;
    } else {
        s.anchor += w;
    }

    return any_hit ? (last_hit_nonce - (start - 1)) : w;
}