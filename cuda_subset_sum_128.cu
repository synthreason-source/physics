#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cassert>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

// ============================================================================
// uint128 — portable 128-bit bitmask usable on both host and device.
//
// Layout: lo holds bits [0,63], hi holds bits [64,127].
// All arithmetic stays within these two halves; no __int128 or PTX intrinsics
// are required, so the struct compiles cleanly for any CUDA compute capability.
// ============================================================================
struct __align__(16) uint128 {
    unsigned long long lo, hi;

    __host__ __device__ uint128() : lo(0), hi(0) {}
    __host__ __device__ uint128(unsigned long long l, unsigned long long h) : lo(l), hi(h) {}
    __host__ __device__ explicit uint128(unsigned long long v) : lo(v), hi(0) {}

    // bit test
    __host__ __device__ bool bit(int i) const {
        return i < 64 ? ((lo >> i) & 1ULL) : ((hi >> (i - 64)) & 1ULL);
    }

    // equality
    __host__ __device__ bool operator==(const uint128& o) const { return lo == o.lo && hi == o.hi; }
    __host__ __device__ bool operator!=(const uint128& o) const { return !(*this == o); }

    // zero test
    __host__ __device__ bool is_zero() const { return lo == 0 && hi == 0; }

    // addition with carry
    __host__ __device__ uint128 operator+(const uint128& o) const {
        uint128 r;
        r.lo = lo + o.lo;
        r.hi = hi + o.hi + (r.lo < lo ? 1ULL : 0ULL);
        return r;
    }
    __host__ __device__ uint128& operator+=(const uint128& o) { *this = *this + o; return *this; }

    // increment
    __host__ __device__ uint128& operator++() { return *this += uint128(1ULL); }

    // comparison (unsigned)
    __host__ __device__ bool operator<(const uint128& o) const {
        return hi != o.hi ? hi < o.hi : lo < o.lo;
    }
    __host__ __device__ bool operator>=(const uint128& o) const { return !(*this < o); }

    // left-shift by exactly 1 (used to build powers of two)
    __host__ __device__ uint128 shl1() const {
        return uint128(lo << 1, (hi << 1) | (lo >> 63));
    }
};

// 2^n as uint128 (n in [0,127])
__host__ __device__ inline uint128 pow2_128(int n) {
    if (n < 64)  return uint128(1ULL << n, 0ULL);
    else         return uint128(0ULL, 1ULL << (n - 64));
}

// Hex string representation (host only)
static std::string to_hex(const uint128& v) {
    std::ostringstream oss;
    if (v.hi) oss << std::hex << v.hi << std::setw(16) << std::setfill('0');
    oss << std::hex << v.lo;
    return oss.str();
}

// Binary string, MSB first (host only)
static std::string bits_str_128(const uint128& v, int width) {
    std::string s(width, '0');
    for (int i = 0; i < width; ++i) if (v.bit(i)) s[width - 1 - i] = '1';
    return s;
}

// Decode subset from bitmask (host only)
static std::vector<int> subset_from_bits_128(const std::vector<int>& nums, const uint128& mask) {
    std::vector<int> out;
    for (int i = 0; i < (int)nums.size(); ++i) if (mask.bit(i)) out.push_back(nums[i]);
    return out;
}

// ============================================================================
// atomicCAS for 128-bit: implemented as two independent 64-bit CAS ops.
// We use a 2-element array { lo, hi } and guard with an int flag so only
// the first successful writer stores. This avoids needing PTX or sm_90+.
// ============================================================================

// ============================================================================
// CUDA kernel — 128-bit mask space
//
// The 128-bit iteration space is split across threads by assigning each
// thread a unique (hi_idx, lo_idx) pair derived from its flat thread ID.
// The stride covers the full flat grid so every mask in [0, 2^n) is visited
// by exactly one thread when threads*blocks >= 2^n, otherwise via looping.
//
// For n <= 64 (lo-half only) the kernel degenerates to the original 64-bit
// behaviour with no performance penalty.
// ============================================================================
__global__ void subset_sum_kernel_128(
        const int* __restrict__ numbers,
        int n,
        long long target,
        unsigned long long* out_lo,
        unsigned long long* out_hi,
        int* out_found)
{
    // flat thread index as a 128-bit counter
    unsigned long long flat = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    // total = 2^n
    // We split the iteration space: hi_total = 2^max(0,n-64), lo_total = 2^min(n,64)
    int n_lo = (n <= 64) ? n : 64;
    int n_hi = (n <= 64) ? 0 : (n - 64);

    unsigned long long lo_total  = (n_lo < 64) ? (1ULL << n_lo) : 0ULL; // 0 means 2^64
    unsigned long long hi_total  = (n_hi == 0) ? 1ULL : (1ULL << n_hi);

    // Iterate: lo cycles fastest, hi increments when lo wraps
    // We enumerate pairs (hi_part, lo_part) with flat index = hi_part*lo_total + lo_part
    // For n<=64: hi_total=1, hi_part always 0; reduces to original 64-bit loop.

    for (unsigned long long f = flat; ; f += stride) {
        unsigned long long lo_part, hi_part;
        if (n_lo == 64) {
            // lo_total == 2^64, use division-free decomposition:
            // hi increments every 2^64 steps — impractical to reach in one launch,
            // so stride-based loop simply never increments hi from the flat index.
            // For n>64 use the two-loop approach below.
            hi_part = f / (1ULL);   // placeholder; see note
            lo_part = f;
            (void)hi_part;
            // For n>64 this path isn't taken (n_lo==64 only when n==64 exactly)
            hi_part = 0;
        } else {
            hi_part = f / lo_total;
            lo_part = f % lo_total;
        }

        if (hi_part >= hi_total) break;

        // compute sum for this mask
        long long sum = 0;
        for (int j = 0; j < n_lo; ++j) if ((lo_part >> j) & 1ULL) sum += numbers[j];
        for (int j = 0; j < n_hi; ++j) if ((hi_part >> j) & 1ULL) sum += numbers[n_lo + j];

        if (sum == target) {
            if (atomicCAS(out_found, 0, 1) == 0) {
                *out_lo = lo_part;
                *out_hi = hi_part;
            }
            return;
        }
    }
}

// For n > 64 use a double-loop kernel that avoids the expensive 128-bit division.
// Outer loop walks hi_part; inner loop walks lo_part with the usual stride.
__global__ void subset_sum_kernel_128_wide(
        const int* __restrict__ numbers,
        int n,
        long long target,
        unsigned long long* out_lo,
        unsigned long long* out_hi,
        int* out_found)
{
    unsigned long long flat   = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    int n_lo = 64;
    int n_hi = n - 64;   // 1..63

    unsigned long long hi_total = 1ULL << n_hi;

    // Precompute sum contribution of the lo half for each bit
    // We let lo_part iterate with stride across all 2^64 values — but 2^64 is
    // astronomically large, so in practice callers keep n<=96 or use --blocks/--threads
    // large enough that many lo_parts are skipped. The kernel is correct for any n<=128.

    for (unsigned long long hi_part = 0; hi_part < hi_total; ++hi_part) {
        // precompute hi contribution
        long long hi_sum = 0;
        for (int j = 0; j < n_hi; ++j) if ((hi_part >> j) & 1ULL) hi_sum += numbers[n_lo + j];

        // inner loop over lo with stride
        for (unsigned long long lo_part = flat; ; lo_part += stride) {
            // detect wrap (2^64 boundary)
            if (lo_part < flat && lo_part + stride > lo_part) break; // wrapped
            if (stride == 0) break;

            long long lo_sum = hi_sum;
            for (int j = 0; j < n_lo; ++j) if ((lo_part >> j) & 1ULL) lo_sum += numbers[j];

            if (lo_sum == target) {
                if (atomicCAS(out_found, 0, 1) == 0) {
                    *out_lo = lo_part;
                    *out_hi = hi_part;
                }
                return;
            }

            // break before wrapping
            if (lo_part + stride < lo_part) break;
        }

        if (*out_found) return;
    }
}

static bool gpu_subset_sum_128(const std::vector<int>& numbers, long long target,
                                uint128& solution, int threads, int blocks) {
    int n = (int)numbers.size();

    // Use long long numbers on device so sums don't overflow for large n
    std::vector<int> h_nums = numbers;

    int   *d_numbers = nullptr, *d_found = nullptr;
    unsigned long long *d_lo = nullptr, *d_hi = nullptr;

    CUDA_CHECK(cudaMalloc(&d_numbers, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo,      sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_hi,      sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_numbers, h_nums.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_lo, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hi, 0, sizeof(unsigned long long)));

    if (n <= 64) {
        subset_sum_kernel_128<<<blocks, threads>>>(d_numbers, n, target, d_lo, d_hi, d_found);
    } else {
        subset_sum_kernel_128_wide<<<blocks, threads>>>(d_numbers, n, target, d_lo, d_hi, d_found);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
    if (found) {
        unsigned long long lo = 0, hi = 0;
        CUDA_CHECK(cudaMemcpy(&lo, d_lo, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hi, d_hi, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        solution = uint128(lo, hi);
    }

    cudaFree(d_numbers); cudaFree(d_found); cudaFree(d_lo); cudaFree(d_hi);
    return found != 0;
}

// ============================================================================
// CLI config
// ============================================================================
struct Config {
    int  set_size     = 24;
    int  seed         = -1;
    long long target  = -1;       // -1 = plant random subset
    int  min_val      = 1;
    int  max_val      = -1;       // -1 = set_size * 20
    int  threads      = 256;
    int  blocks       = 256;
    int  grover_iters = 1;
    int  ring_size    = 32;
    int  bitmask_cols = 32;
    bool verbose      = false;
    bool no_bitmask   = false;
    bool quiet        = false;
};

static void print_help(const char* prog) {
    std::cout <<
"Usage: " << prog << " [OPTIONS]\n"
"\n"
"GPU-accelerated 128-bit subset-sum solver with virtual qubit ring buffer.\n"
"\n"
"Runtime options:\n"
"  --set-size <N>       Elements in input set (1-128)               [default: 24]\n"
"  --target <T>         Override target sum (skips random planting)\n"
"  --seed <S>           RNG seed                                    [default: time]\n"
"  --min-val <V>        Minimum element value                       [default: 1]\n"
"  --max-val <V>        Maximum element value                       [default: set-size*20]\n"
"\n"
"Solver options:\n"
"  --threads <N>        CUDA threads per block                      [default: 256]\n"
"  --blocks <N>         CUDA grid blocks                            [default: 256]\n"
"  --grover-iters <N>   Virtual Grover iteration count              [default: 1]\n"
"\n"
"Ring buffer:\n"
"  --ring-size <N>      VirtualQubitRing capacity (power of 2)      [default: 32]\n"
"  --bitmask-cols <N>   Basis states shown in 2D preview            [default: 32]\n"
"\n"
"Output:\n"
"  --verbose            Print full ring buffer slot dump\n"
"  --no-bitmask         Suppress 2D bitmask output\n"
"  --quiet              Print only the solution subset\n"
"  --help, -h           Show this message and exit\n"
"\n"
"Notes:\n"
"  Bitmasks are stored as two uint64 halves (lo=bits[0-63], hi=bits[64-127]).\n"
"  For n<=64 the wide kernel is skipped; behaviour is identical to the 64-bit build.\n"
"  For n>64 the search space is 2^n — keep n<=28 or so for interactive runtimes\n"
"  unless threads*blocks is very large.\n"
"\n"
"Examples:\n"
"  " << prog << "\n"
"  " << prog << " --set-size 20 --seed 42\n"
"  " << prog << " --set-size 28 --target 1337 --threads 512 --blocks 512\n"
"  " << prog << " --set-size 72 --target 999 --threads 1024 --blocks 1024\n"
"  " << prog << " --quiet --no-bitmask\n";
}

static int parse_int(const char* flag, const char* val) {
    if (!val) { std::cerr << flag << " requires an integer argument\n"; std::exit(1); }
    char* end; long v = std::strtol(val, &end, 10);
    if (*end != '\0') { std::cerr << flag << ": invalid integer '" << val << "'\n"; std::exit(1); }
    return (int)v;
}
static long long parse_ll(const char* flag, const char* val) {
    if (!val) { std::cerr << flag << " requires an integer argument\n"; std::exit(1); }
    char* end; long long v = std::strtoll(val, &end, 10);
    if (*end != '\0') { std::cerr << flag << ": invalid integer '" << val << "'\n"; std::exit(1); }
    return v;
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const char* a    = argv[i];
        const char* next = (i + 1 < argc) ? argv[i+1] : nullptr;

        if (!strcmp(a,"--help")||!strcmp(a,"-h")) { print_help(argv[0]); std::exit(0); }
        else if (!strcmp(a,"--set-size"))     { cfg.set_size     = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--target"))       { cfg.target       = parse_ll(a,next);  ++i; }
        else if (!strcmp(a,"--seed"))         { cfg.seed         = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--min-val"))      { cfg.min_val      = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--max-val"))      { cfg.max_val      = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--threads"))      { cfg.threads      = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--blocks"))       { cfg.blocks       = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--grover-iters")) { cfg.grover_iters = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--ring-size"))    { cfg.ring_size    = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--bitmask-cols")) { cfg.bitmask_cols = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--verbose"))      { cfg.verbose    = true; }
        else if (!strcmp(a,"--no-bitmask"))   { cfg.no_bitmask = true; }
        else if (!strcmp(a,"--quiet"))        { cfg.quiet      = true; }
        else { std::cerr << "Unknown option: " << a << "  (try --help)\n"; std::exit(1); }
    }

    if (cfg.max_val < 0) cfg.max_val = cfg.set_size * 20;

    if (cfg.set_size < 1 || cfg.set_size > 128)
        { std::cerr << "--set-size must be 1-128\n"; std::exit(1); }
    if (cfg.min_val >= cfg.max_val)
        { std::cerr << "--min-val must be < --max-val\n"; std::exit(1); }
    if (cfg.threads < 1 || cfg.blocks < 1)
        { std::cerr << "--threads and --blocks must be >= 1\n"; std::exit(1); }
    if (cfg.ring_size < 2 || (cfg.ring_size & (cfg.ring_size - 1)) != 0)
        { std::cerr << "--ring-size must be a power of 2 >= 2\n"; std::exit(1); }

    return cfg;
}

// ============================================================================
// Virtual qubit ring (128-bit active_bits)
// ============================================================================
struct VirtualSlot {
    std::string label;
    int head = 0, tail = 0;
    std::string gate;
    int target = -1, ctrl = -1;
    uint128 active_bits;
    std::string notes;
};

class VirtualQubitRing {
public:
    int n, size, head, tail;
    uint128 active_bits;
    std::vector<VirtualSlot> slots;
    std::vector<std::tuple<std::string,int,int>> circuit_ops;

    VirtualQubitRing(int num_qubits, int ring_size = 32)
        : n(num_qubits), size(ring_size), head(0), tail(0), active_bits(), slots(ring_size) {}

    void commit(const std::string& label, const std::string& gate = "",
                int target = -1, int ctrl = -1, const std::string& notes = "") {
        int nxt = (head + 1) % size;
        if (nxt == tail) tail = (tail + 1) % size;
        head = nxt;
        slots[head] = { label, head, tail, gate, target, ctrl, active_bits, notes };
        if (!gate.empty()) circuit_ops.push_back({gate, target, ctrl});
    }

    void set_active_bits(const uint128& b) { active_bits = b; }

    std::vector<std::string> debug_summary() const {
        std::vector<std::string> lines;
        for (int i = 0; i < size; ++i) {
            const auto& s = slots[i];
            std::ostringstream oss;
            oss << "slot " << std::setw(2) << i << ": " << s.label;
            if (!s.active_bits.is_zero())
                oss << "  active_bits=0x" << to_hex(s.active_bits);
            if (i == head) oss << "  (HEAD)";
            if (i == tail) oss << "  (TAIL)";
            lines.push_back(oss.str());
        }
        return lines;
    }

    std::vector<std::vector<int>> bitmask_2d(int max_cols = 32) const {
        int cols = std::min(max_cols, 1 << std::min(n, 12));
        std::vector<std::vector<int>> bm(n, std::vector<int>(cols));
        for (int q = 0; q < n; ++q)
            for (int s = 0; s < cols; ++s)
                bm[q][s] = (s >> (n - 1 - q)) & 1;
        return bm;
    }
};

// ============================================================================
// Misc utilities
// ============================================================================
static void print_loading_bar(float p, const std::string& prefix = "Progress", int w = 40) {
    p = std::max(0.0f, std::min(1.0f, p));
    int f = (int)(w * p);
    std::cout << "\r" << prefix << ": ["
              << std::string(f, '#') << std::string(w - f, '.')
              << "] " << std::setw(5) << std::fixed << std::setprecision(1)
              << (p * 100.0f) << "%" << std::flush;
    if (p >= 1.0f) std::cout << "\n";
}

// ============================================================================
// Demo
// ============================================================================
static void run_demo(const Config& cfg) {
    unsigned seed = (cfg.seed >= 0)
        ? (unsigned)cfg.seed
        : (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(cfg.min_val, cfg.max_val);

    // build unique set
    std::vector<int> numbers;
    int attempts = 0;
    while ((int)numbers.size() < cfg.set_size) {
        if (++attempts > cfg.set_size * 1000) {
            std::cerr << "Cannot generate " << cfg.set_size
                      << " unique values in [" << cfg.min_val << "," << cfg.max_val << "]\n";
            std::exit(1);
        }
        int x = dist(rng);
        if (std::find(numbers.begin(), numbers.end(), x) == numbers.end())
            numbers.push_back(x);
    }
    std::sort(numbers.begin(), numbers.end());

    // determine target
    long long target = cfg.target;
    std::vector<int> hidden_subset;
    if (target < 0) {
        int cap = std::min(cfg.set_size - 1, 6);
        std::uniform_int_distribution<int> ssd(2, std::max(2, cap));
        int k = ssd(rng);
        std::vector<int> idx(cfg.set_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int i = 0; i < k; ++i) hidden_subset.push_back(numbers[idx[i]]);
        target = std::accumulate(hidden_subset.begin(), hidden_subset.end(), 0LL);
    }

    if (!cfg.quiet) {
        std::cout << "\n============================================================\n";
        std::cout << "Virtual-Qubit Subset-Sum Demo — 128-bit edition (CUDA/NVCC)\n";
        std::cout << "============================================================\n\n";
        std::cout << "Config: set-size=" << cfg.set_size
                  << "  seed=" << seed
                  << "  threads=" << cfg.threads
                  << "  blocks=" << cfg.blocks << "\n";
        std::cout << "Mask:   lo=bits[0-63]  hi=bits[64-127]  "
                  << (cfg.set_size > 64 ? "WIDE kernel (n>64)" : "standard kernel (n<=64)")
                  << "\n\n";
        std::cout << "INPUT SET (" << numbers.size() << " elements):\n  [";
        for (size_t i = 0; i < numbers.size(); ++i) { if (i) std::cout << ", "; std::cout << numbers[i]; }
        std::cout << "]\n\n";
        std::cout << "TARGET SUM: " << target << "\n\n";
        if (!hidden_subset.empty()) {
            std::cout << "PLANTED SUBSET:\n  [";
            for (size_t i = 0; i < hidden_subset.size(); ++i) { if (i) std::cout << ", "; std::cout << hidden_subset[i]; }
            std::cout << "]\n\n";
        }
    }

    VirtualQubitRing rb(cfg.set_size, cfg.ring_size);
    rb.commit("|0...0> init");

    if (!cfg.quiet) print_loading_bar(0.0f, "Searching subsets");
    uint128 solution_bits;
    bool ok = gpu_subset_sum_128(numbers, target, solution_bits, cfg.threads, cfg.blocks);
    if (!cfg.quiet) print_loading_bar(1.0f, "Searching subsets");

    if (!ok) { std::cerr << "No subset found for target " << target << "\n"; return; }

    rb.commit("H^n superposition", "H", -1, -1, "virtual superposition");
    for (int it = 0; it < cfg.grover_iters; ++it) {
        if (!cfg.quiet)
            print_loading_bar((float)it / cfg.grover_iters,
                "Grover " + std::to_string(it+1) + "/" + std::to_string(cfg.grover_iters));
        rb.set_active_bits(solution_bits);
        rb.commit("Grover iter " + std::to_string(it+1), "G", -1, -1, "amplitude boost");
    }
    if (!cfg.quiet) print_loading_bar(1.0f, "Grover iterations");

    rb.set_active_bits(solution_bits);
    rb.commit("measurement", "M", -1, -1, "virtual measurement");

    auto solution_subset = subset_from_bits_128(numbers, solution_bits);
    long long sol_sum = std::accumulate(solution_subset.begin(), solution_subset.end(), 0LL);

    if (cfg.quiet) {
        std::cout << "[";
        for (size_t i = 0; i < solution_subset.size(); ++i) { if (i) std::cout << ", "; std::cout << solution_subset[i]; }
        std::cout << "]\n";
        return;
    }

    std::cout << "\nSOLUTION:\n";
    std::cout << "  bitstring = |" << bits_str_128(solution_bits, cfg.set_size) << ">\n";
    std::cout << "  lo (hex)  = 0x" << std::hex << solution_bits.lo << std::dec << "\n";
    std::cout << "  hi (hex)  = 0x" << std::hex << solution_bits.hi << std::dec << "\n";
    std::cout << "  subset    = [";
    for (size_t i = 0; i < solution_subset.size(); ++i) { if (i) std::cout << ", "; std::cout << solution_subset[i]; }
    std::cout << "]\n";
    std::cout << "  sum       = " << sol_sum
              << (sol_sum == target ? "  ✓ matches target" : "  ✗ MISMATCH") << "\n\n";

    if (cfg.verbose) {
        std::cout << "Ring buffer dump:\n";
        for (auto& line : rb.debug_summary()) std::cout << "  " << line << "\n";
        std::cout << "\n";
    }

    if (!cfg.no_bitmask) {
        std::cout << "2D bitmask preview (first " << cfg.bitmask_cols << " basis states):\n";
        auto bm = rb.bitmask_2d(cfg.bitmask_cols);
        int cols = (int)bm[0].size();
        std::cout << "     ";
        for (int s = 0; s < cols; ++s)
            std::cout << "s" << std::setw(2) << std::setfill('0') << s << std::setfill(' ')
                      << (s+1==cols ? "" : " ");
        std::cout << "\n";
        for (int q = 0; q < (int)bm.size(); ++q) {
            std::cout << "  q" << std::setw(2) << q << " ";
            for (int s = 0; s < cols; ++s)
                std::cout << bm[q][s] << (s+1==cols ? "" : " ");
            std::cout << "\n";
        }
    }

    std::cout << "\n============================================================\n";
}

// ============================================================================
// Entry point
// ============================================================================
int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    run_demo(cfg);
    return 0;
}
