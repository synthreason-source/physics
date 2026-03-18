#pragma once
#include <Arduino.h>
#include <ArduinoJson.h>    // StaticJsonDocument
#include <map>              // std::map
#include <deque>            // std::deque
#include <vector>           // std::vector (merkle_branch)
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "pool.h"           // PoolClass, pool_info_t
#include "ecdf_miner.h"     // ecdf_state_t, ecdf_result_t

// ── Compile-time constants used by thread_entry.cpp ──────────────────────────
#ifndef HELLO_POOL_INTERVAL_MS
#  define HELLO_POOL_INTERVAL_MS    (30 * 1000)
#endif
#ifndef POOL_INACTIVITY_TIME_MS
#  define POOL_INACTIVITY_TIME_MS   (120 * 1000)
#endif

// ── Stratum down-stream message types ─────────────────────────────────────────
typedef enum {
    STRATUM_DOWN_PARSE_ERROR      = -1,
    STRATUM_DOWN_SUCCESS          =  0,
    STRATUM_DOWN_ERROR            =  1,
    STRATUM_DOWN_NOTIFY           =  2,
    STRATUM_DOWN_SET_DIFFICULTY   =  3,
    STRATUM_DOWN_SET_VERSION_MASK =  4,
    STRATUM_DOWN_SET_EXTRANONCE   =  5,
    STRATUM_DOWN_UNKNOWN          =  6,
} stratum_down_t;

// ── Stratum subscription info ─────────────────────────────────────────────────
typedef struct {
    String extranonce1;
    String extranonce2;
    int    extranonce2_size;
} stratum_sub_info_t;

// ── Stratum configuration ─────────────────────────────────────────────────────
typedef struct {
    String user;
    String pwd;
} stratum_info_t;

// ── MerkleBranch ─────────────────────────────────────────────────────────────
// Wraps std::vector<String> with an operator= that accepts a JsonVariant so
// that thread_entry.cpp can do: job.merkle_branch = json["params"][4];
struct MerkleBranch : public std::vector<String> {
    using std::vector<String>::vector;
    using std::vector<String>::operator=;

    // Accept any ArduinoJson proxy / variant (JsonArray, ElementProxy, etc.)
    MerkleBranch& operator=(JsonVariantConst arr) {
        this->clear();
        for (JsonVariantConst v : arr.as<JsonArrayConst>())
            this->push_back(v.as<String>());
        return *this;
    }
    // Non-const variant needed for certain ArduinoJson 6 proxy types
    template<typename T>
    MerkleBranch& operator=(const T& arr) {
        this->clear();
        JsonVariantConst cv = arr;
        for (JsonVariantConst v : cv.as<JsonArrayConst>())
            this->push_back(v.as<String>());
        return *this;
    }
};

// ── Pool job data ─────────────────────────────────────────────────────────────
typedef struct {
    String       id;
    String       prevhash;
    String       coinb1;
    String       coinb2;
    String       version;
    String       nbits;
    String       ntime;
    bool         clean_jobs;
    MerkleBranch merkle_branch;  // supports: job.merkle_branch = json["params"][4]
} pool_job_data_t;

// ── Per-message response record ───────────────────────────────────────────────
// thread_entry.cpp accesses rsp.stamp; we keep both ts_ms and stamp.
typedef struct {
    String   method;
    bool     status;
    uint32_t ts_ms;   // timestamp ms when request was sent
    uint32_t stamp;   // same value; alias used by thread_entry.cpp
} stratum_rsp;

// ── Downstream method data returned by listen_methods() ──────────────────────
typedef struct {
    int32_t        id;
    stratum_down_t type;
    String         method;
    String         raw;
} stratum_method_data;

// ─────────────────────────────────────────────────────────────────────────────
// StratumClass
// ─────────────────────────────────────────────────────────────────────────────
class StratumClass {
public:
    // ── Pool handle ───────────────────────────────────────────────────────────
    PoolClass *pool = nullptr;

    // ── FreeRTOS semaphores (accessed directly by thread_entry.cpp) ──────────
    SemaphoreHandle_t new_job_xsem   = nullptr;
    SemaphoreHandle_t clear_job_xsem = nullptr;

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    StratumClass() {
        new_job_xsem   = xSemaphoreCreateBinary();
        clear_job_xsem = xSemaphoreCreateBinary();
    }

    StratumClass(pool_info_t pConfig, stratum_info_t sConfig, size_t cacheSize = 8)
        : _stratum_info(sConfig), _pool_job_cache_size(cacheSize)
    {
        this->pool     = new PoolClass(pConfig);
        new_job_xsem   = xSemaphoreCreateBinary();
        clear_job_xsem = xSemaphoreCreateBinary();
    }

    ~StratumClass();

    void reset();
    void reset(pool_info_t pConfig, stratum_info_t sConfig);

    // ── Stratum protocol ──────────────────────────────────────────────────────
    bool subscribe();
    bool authorize();
    bool suggest_difficulty();
    bool config_version_rolling();
    bool hello_pool(uint32_t hello_interval, uint32_t lost_max_time);

    bool submit(String pool_job_id, String extranonce2,
                uint32_t ntime, uint32_t nonce, uint32_t version);

    stratum_method_data listen_methods();

    // ── Extranonce helpers ────────────────────────────────────────────────────
    String get_sub_extranonce1();
    String get_sub_extranonce2();
    bool   clear_sub_extranonce2();
    void   set_sub_extranonce1(String extranonce1);
    void   set_sub_extranonce2_size(int size);

    // ── Message-response map ──────────────────────────────────────────────────
    bool        set_msg_rsp_map(uint32_t id, bool status);
    bool        del_msg_rsp_map(uint32_t id);
    stratum_rsp get_method_rsp_by_id(uint32_t id);

    // ── Job cache ─────────────────────────────────────────────────────────────
    size_t          push_job_cache(pool_job_data_t job);
    pool_job_data_t pop_job_cache();
    size_t          get_job_cache_size();
    size_t          clear_job_cache();

    // ── Diagnostics ───────────────────────────────────────────────────────────
    bool   is_submit_timeout();
    bool   is_primary_pool_available(String url, uint16_t port);

    // ── Getters / setters ─────────────────────────────────────────────────────
    inline bool     is_subscribed()             const { return _is_subscribed; }
    inline bool     is_authorized()             const { return _is_authorized; }

    inline double   get_pool_difficulty()       const { return _pool_difficulty; }
    inline void     set_pool_difficulty(double d)     { _pool_difficulty = d; }

    inline uint32_t vr_mask()                   const { return _vr_mask; }
    inline void     set_vr_mask(uint32_t m)           { _vr_mask = m; }

    // Alias used by thread_entry.cpp
    inline void     set_version_mask(uint32_t m)      { _vr_mask = m; }

    // thread_entry.cpp: set_authorize(json["result"]) and set_authorize(false)

    template<typename T>
    inline void     set_authorize(T v) { _is_authorized = (bool)v; }

    // thread_entry.cpp: job_counter_inc() / get_job_counter()
    inline void     job_counter_inc()                  { _job_counter++; }
    inline uint32_t get_job_counter()           const  { return _job_counter; }

    // ── ECDF miner ────────────────────────────────────────────────────────────
    uint32_t      ecdf_do_calibrate(uint32_t budget,
                                    bool   (*is_valid_fn)(uint32_t nonce));
    ecdf_result_t ecdf_next_window(float  conf,
                                   bool (*is_valid_fn)(uint32_t nonce));
    uint32_t      ecdf_run(float conf, uint32_t target_hits, uint32_t max_hashes,
                           bool (*is_valid_fn)(uint32_t nonce),
                           void (*on_hit_fn)(uint32_t nonce));
    bool          ecdf_is_fitted()              const;
    float         ecdf_mean_gap()               const;
    uint32_t      ecdf_budget_for_conf(float p) const;
    float         ecdf_conf_for_budget(uint32_t budget) const;
    float         ecdf_marginal_gain(uint32_t budget)   const;

private:
    stratum_info_t     _stratum_info         = {};
    stratum_sub_info_t _sub_info             = {"", "0", 0};
    bool               _is_subscribed        = false;
    bool               _is_authorized        = false;
    double             _pool_difficulty      = 0.0;
    uint32_t           _vr_mask             = 0xffffffff;
    bool               _suggest_diff_support = true;
    uint32_t           _gid                 = 1;

    String                   _rsp_str;
    StaticJsonDocument<4096> _rsp_json;

    std::map<uint32_t, stratum_rsp> _msg_rsp_map;
    static constexpr size_t         _max_rsp_id_cache = 32;

    std::deque<pool_job_data_t> _pool_job_cache;
    size_t                      _pool_job_cache_size = 8;

    uint32_t     _job_counter = 0;
    ecdf_state_t _ecdf        = {};

    uint32_t _get_msg_id();
    bool     _parse_rsp();
    bool     _clear_rsp_id_cache();
};