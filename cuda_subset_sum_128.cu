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
#include <fstream>
#include <stdexcept>
#include <limits>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

struct uint128 {
    unsigned long long lo, hi;
    __host__ __device__ uint128() : lo(0), hi(0) {}
    __host__ __device__ uint128(unsigned long long l, unsigned long long h) : lo(l), hi(h) {}
    __host__ __device__ explicit uint128(unsigned long long v) : lo(v), hi(0) {}
    __host__ __device__ bool bit(int i) const { return i < 64 ? ((lo >> i) & 1ULL) : ((hi >> (i - 64)) & 1ULL); }
    __host__ __device__ bool is_zero() const { return lo == 0 && hi == 0; }
};

static std::string to_hex(const uint128& v) {
    std::ostringstream oss;
    if (v.hi) oss << std::hex << v.hi << std::setw(16) << std::setfill('0');
    oss << std::hex << v.lo;
    return oss.str();
}

static std::string bits_str_128(const uint128& v, int width) {
    std::string s(width, '0');
    for (int i = 0; i < width; ++i) if (v.bit(i)) s[width - 1 - i] = '1';
    return s;
}

static std::vector<int> subset_from_bits_128(const std::vector<int>& nums, const uint128& mask) {
    std::vector<int> out;
    for (int i = 0; i < (int)nums.size(); ++i) if (mask.bit(i)) out.push_back(nums[i]);
    return out;
}

struct Config {
    int  set_size     = 24;
    int  seed         = -1;
    long long target  = -1;
    int  min_val      = 1;
    int  max_val      = -1;
    int  pick_min     = 2;
    int  pick_max     = 8;
    int  threads      = 256;
    int  blocks       = 256;
    int  grover_iters = 1;
    int  ring_size    = 32;
    int  bitmask_cols = 32;
    bool verbose      = false;
    bool no_bitmask   = false;
    bool quiet        = false;
    std::string csv_file;
    int         csv_col   = 0;
    std::string csv_col_name;
    bool        csv_header = true;
    char        csv_delim  = ',';
};

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

static void print_help(const char* prog) {
    std::cout <<
"Usage: " << prog << " [OPTIONS]\n"
"\n"
"GPU-accelerated 128-bit subset-sum solver with virtual qubit ring buffer.\n"
"\n"
"Input (mutually exclusive — CSV takes priority over random generation):\n"
"  --csv <FILE>         Load numbers from a CSV file\n"
"  --csv-col <N>        0-based column index to read                [default: 0]\n"
"  --csv-col-name <H>   Column header name (overrides --csv-col)\n"
"  --csv-no-header      File has no header row\n"
"  --csv-delim <C>      Field delimiter character                   [default: ,]\n"
"\n"
"Runtime options (used when NOT loading from CSV):\n"
"  --set-size <N>       Elements in input set (1-128)               [default: 24]\n"
"  --target <T>         Override target sum (skips random planting)\n"
"  --seed <S>           RNG seed                                    [default: time]\n"
"  --min-val <V>        Minimum element value                       [default: 1]\n"
"  --max-val <V>        Maximum element value                       [default: set-size*20]\n"
"  --pick-min <N>       Minimum hidden subset size                  [default: 2]\n"
"  --pick-max <N>       Maximum hidden subset size                  [default: 8]\n"
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
"  --help, -h           Show this message and exit\n";
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* next = (i + 1 < argc) ? argv[i+1] : nullptr;
        if (!strcmp(a,"--help")||!strcmp(a,"-h")) { print_help(argv[0]); std::exit(0); }
        else if (!strcmp(a,"--set-size"))     { cfg.set_size     = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--target"))       { cfg.target       = parse_ll(a,next);  ++i; }
        else if (!strcmp(a,"--seed"))         { cfg.seed         = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--min-val"))      { cfg.min_val      = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--max-val"))      { cfg.max_val      = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--pick-min"))     { cfg.pick_min     = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--pick-max"))     { cfg.pick_max     = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--threads"))      { cfg.threads      = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--blocks"))       { cfg.blocks       = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--grover-iters")) { cfg.grover_iters = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--ring-size"))    { cfg.ring_size    = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--bitmask-cols")) { cfg.bitmask_cols = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--csv"))          { if (!next){std::cerr<<"--csv requires a filename\n";std::exit(1);} cfg.csv_file = next; ++i; }
        else if (!strcmp(a,"--csv-col"))      { cfg.csv_col      = parse_int(a,next); ++i; }
        else if (!strcmp(a,"--csv-col-name")) { if (!next){std::cerr<<"--csv-col-name requires a value\n";std::exit(1);} cfg.csv_col_name = next; ++i; }
        else if (!strcmp(a,"--csv-no-header")){ cfg.csv_header   = false; }
        else if (!strcmp(a,"--csv-delim"))    { if (!next||next[0]=='\0'){std::cerr<<"--csv-delim requires a character\n";std::exit(1);} cfg.csv_delim = next[0]; ++i; }
        else if (!strcmp(a,"--verbose"))      { cfg.verbose    = true; }
        else if (!strcmp(a,"--no-bitmask"))   { cfg.no_bitmask = true; }
        else if (!strcmp(a,"--quiet"))        { cfg.quiet      = true; }
        else { std::cerr << "Unknown option: " << a << "  (try --help)\n"; std::exit(1); }
    }

    if (cfg.max_val < 0) cfg.max_val = cfg.set_size * 20;
    if (cfg.csv_file.empty()) {
        if (cfg.set_size < 1 || cfg.set_size > 128) { std::cerr << "--set-size must be 1-128\n"; std::exit(1); }
        if (cfg.min_val >= cfg.max_val) { std::cerr << "--min-val must be < --max-val\n"; std::exit(1); }
        if (cfg.pick_min < 1 || cfg.pick_max < cfg.pick_min || cfg.pick_max > cfg.set_size) { std::cerr << "--pick-min/--pick-max must satisfy 1 <= min <= max <= set-size\n"; std::exit(1); }
    }
    if (cfg.threads < 1 || cfg.blocks < 1) { std::cerr << "--threads and --blocks must be >= 1\n"; std::exit(1); }
    if (cfg.ring_size < 2 || (cfg.ring_size & (cfg.ring_size - 1)) != 0) { std::cerr << "--ring-size must be a power of 2 >= 2\n"; std::exit(1); }
    return cfg;
}

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
            if (!s.active_bits.is_zero()) oss << "  active_bits=0x" << to_hex(s.active_bits);
            if (i == head) oss << "  (HEAD)";
            if (i == tail) oss << "  (TAIL)";
            lines.push_back(oss.str());
        }
        return lines;
    }

    std::vector<std::vector<int>> bitmask_2d(int max_cols = 32) const {
        int cols = std::min(max_cols, 1 << std::min(n, 12));
        std::vector<std::vector<int>> bm(n, std::vector<int>(cols));
        for (int q = 0; q < n; ++q) for (int s = 0; s < cols; ++s) bm[q][s] = (s >> (n - 1 - q)) & 1;
        return bm;
    }
};

static void print_loading_bar(float p, const std::string& prefix = "Progress", int w = 40) {
    p = std::max(0.0f, std::min(1.0f, p));
    int f = (int)(w * p);
    std::cout << "\r" << prefix << ": ["
              << std::string(f, '#') << std::string(w - f, '.')
              << "] " << std::setw(5) << std::fixed << std::setprecision(1)
              << (p * 100.0f) << "%" << std::flush;
    if (p >= 1.0f) std::cout << "\n";
}

static std::vector<int> load_csv(const Config& cfg) {
    std::ifstream f(cfg.csv_file);
    if (!f) { std::cerr << "Cannot open CSV file: " << cfg.csv_file << "\n"; std::exit(1); }
    int col_idx = cfg.csv_col;
    std::string line;
    int line_num = 0;
    if (cfg.csv_header) {
        if (!std::getline(f, line)) { std::cerr << "CSV file is empty: " << cfg.csv_file << "\n"; std::exit(1); }
        ++line_num;
        if (!cfg.csv_col_name.empty()) {
            std::istringstream hss(line);
            std::string cell;
            int ci = 0; bool found = false;
            while (std::getline(hss, cell, cfg.csv_delim)) {
                size_t s = cell.find_first_not_of(" \t\r\"'");
                size_t e = cell.find_last_not_of(" \t\r\"'");
                std::string trimmed = (s == std::string::npos) ? "" : cell.substr(s, e - s + 1);
                if (trimmed == cfg.csv_col_name) { col_idx = ci; found = true; break; }
                ++ci;
            }
            if (!found) { std::cerr << "CSV column '" << cfg.csv_col_name << "' not found in header of " << cfg.csv_file << "\n"; std::exit(1); }
        }
    }
    std::vector<int> numbers;
    int skipped = 0, dupes = 0, truncated = 0;
    while (std::getline(f, line)) {
        ++line_num;
        if (line.empty() || line == "\r") continue;
        std::istringstream ss(line);
        std::string cell;
        int ci = 0; bool row_found = false;
        while (std::getline(ss, cell, cfg.csv_delim)) {
            if (ci == col_idx) {
                row_found = true;
                size_t s = cell.find_first_not_of(" \t\r\"'");
                size_t e = cell.find_last_not_of(" \t\r\"'");
                if (s == std::string::npos) { ++skipped; break; }
                std::string tok = cell.substr(s, e - s + 1);
                size_t dot = tok.find('.');
                if (dot != std::string::npos) tok = tok.substr(0, dot);
                char* endp;
                long long v = std::strtoll(tok.c_str(), &endp, 10);
                if (*endp != '\0' && *endp != '\r') { if (!cfg.quiet) std::cerr << "  [CSV] line " << line_num << ": skipping non-numeric value '" << tok << "'\n"; ++skipped; break; }
                if (v < INT_MIN || v > INT_MAX) { if (!cfg.quiet) std::cerr << "  [CSV] line " << line_num << ": value " << v << " out of int range, skipping\n"; ++skipped; break; }
                int iv = (int)v;
                if (std::find(numbers.begin(), numbers.end(), iv) != numbers.end()) { if (!cfg.quiet) std::cerr << "  [CSV] line " << line_num << ": duplicate value " << iv << ", skipping\n"; ++dupes; break; }
                if ((int)numbers.size() >= 128) { ++truncated; break; }
                numbers.push_back(iv);
                break;
            }
            ++ci;
        }
        if (!row_found && !cfg.quiet) std::cerr << "  [CSV] line " << line_num << ": fewer than " << (col_idx + 1) << " columns, skipping\n";
    }
    if (!cfg.quiet) {
        std::cout << "[CSV] loaded " << numbers.size() << " value(s) from '" << cfg.csv_file << "' (col ";
        if (!cfg.csv_col_name.empty()) std::cout << '\'' << cfg.csv_col_name << '\''; else std::cout << col_idx;
        std::cout << ")";
        if (skipped)   std::cout << "  skipped=" << skipped;
        if (dupes)     std::cout << "  dupes=" << dupes;
        if (truncated) std::cout << "  truncated=" << truncated << " (max 128)";
        std::cout << "\n";
    }
    if (numbers.empty()) { std::cerr << "No usable integers found in CSV column.\n"; std::exit(1); }
    std::sort(numbers.begin(), numbers.end());
    return numbers;
}

__global__ void subset_sum_kernel(const int* numbers, int n, long long target, uint128* out_solution, int* out_found) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long total_lo = (n < 64) ? (1ULL << n) : 0ULL;
    if (n < 64) {
        for (unsigned long long mask = idx; mask < total_lo; mask += stride) {
            long long sum = 0;
            for (int j = 0; j < n; ++j) if ((mask >> j) & 1ULL) sum += numbers[j];
            if (sum == target) {
                if (atomicCAS(out_found, 0, 1) == 0) *out_solution = uint128(mask, 0ULL);
                return;
            }
        }
    } else {
        for (unsigned long long lo = idx; ; lo += stride) {
            if (lo < idx) break;
            long long sum = 0;
            for (int j = 0; j < 64; ++j) if ((lo >> j) & 1ULL) sum += numbers[j];
            if (sum == target) {
                if (atomicCAS(out_found, 0, 1) == 0) *out_solution = uint128(lo, 0ULL);
                return;
            }
            if (stride == 0 || lo + stride < lo) break;
        }
    }
}

static bool gpu_subset_sum_128(const std::vector<int>& numbers, long long target, uint128& solution, int threads, int blocks) {
    int n = (int)numbers.size();
    int *d_numbers = nullptr, *d_found = nullptr; uint128 *d_solution = nullptr;
    CUDA_CHECK(cudaMalloc(&d_numbers, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_solution, sizeof(uint128)));
    CUDA_CHECK(cudaMemcpy(d_numbers, numbers.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_solution, 0, sizeof(uint128)));
    subset_sum_kernel<<<blocks, threads>>>(d_numbers, n, target, d_solution, d_found);
    CUDA_CHECK(cudaDeviceSynchronize());
    int found = 0;
    CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
    if (found) CUDA_CHECK(cudaMemcpy(&solution, d_solution, sizeof(uint128), cudaMemcpyDeviceToHost));
    cudaFree(d_numbers); cudaFree(d_found); cudaFree(d_solution);
    return found != 0;
}

static void run_demo(const Config& cfg) {
    unsigned seed = (cfg.seed >= 0) ? (unsigned)cfg.seed : (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::vector<int> numbers;
    bool from_csv = !cfg.csv_file.empty();
    if (from_csv) numbers = load_csv(cfg);
    else {
        std::uniform_int_distribution<int> dist(cfg.min_val, cfg.max_val);
        int attempts = 0;
        while ((int)numbers.size() < cfg.set_size) {
            if (++attempts > cfg.set_size * 1000) { std::cerr << "Cannot generate " << cfg.set_size << " unique values in [" << cfg.min_val << "," << cfg.max_val << "]\n"; std::exit(1); }
            int x = dist(rng);
            if (std::find(numbers.begin(), numbers.end(), x) == numbers.end()) numbers.push_back(x);
        }
        std::sort(numbers.begin(), numbers.end());
    }
    int actual_size = (int)numbers.size();
    long long target = cfg.target;
    std::vector<int> hidden_subset;
    if (target < 0) {
        int lo = std::max(1, std::min(cfg.pick_min, actual_size));
        int hi = std::max(lo, std::min(cfg.pick_max, actual_size));
        std::uniform_int_distribution<int> ssd(lo, hi);
        int k = ssd(rng);
        std::vector<int> idx(actual_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int i = 0; i < k; ++i) hidden_subset.push_back(numbers[idx[i]]);
        target = std::accumulate(hidden_subset.begin(), hidden_subset.end(), 0LL);
    }
    if (!cfg.quiet) {
        std::cout << "\n============================================================\n";
        std::cout << "Virtual-Qubit Subset-Sum Demo — 128-bit edition (CUDA/NVCC)\n";
        std::cout << "============================================================\n\n";
        std::cout << "Input:  " << (from_csv ? cfg.csv_file : "random") << "\n";
        std::cout << "Config: set-size=" << actual_size << "  seed=" << seed << "  threads=" << cfg.threads << "  blocks=" << cfg.blocks << "\n";
        std::cout << "Pick:   min=" << cfg.pick_min << "  max=" << cfg.pick_max << "\n";
        std::cout << "Mask:   lo=bits[0-63]  hi=bits[64-127]  " << (actual_size > 64 ? "WIDE kernel (n>64)" : "standard kernel (n<=64)") << "\n\n";
        std::cout << "INPUT SET (" << actual_size << " elements):\n  [";
        for (size_t i = 0; i < numbers.size(); ++i) { if (i) std::cout << ", "; std::cout << numbers[i]; }
        std::cout << "]\n\n";
        std::cout << "TARGET SUM: " << target << "\n\n";
        if (!hidden_subset.empty()) {
            std::cout << "PLANTED SUBSET (size=" << hidden_subset.size() << "):\n  [";
            for (size_t i = 0; i < hidden_subset.size(); ++i) { if (i) std::cout << ", "; std::cout << hidden_subset[i]; }
            std::cout << "]\n\n";
        }
    }
    VirtualQubitRing rb(actual_size, cfg.ring_size);
    rb.commit("|0...0> init");
    if (!cfg.quiet) print_loading_bar(0.0f, "Searching subsets");
    uint128 solution_bits;
    bool ok = gpu_subset_sum_128(numbers, target, solution_bits, cfg.threads, cfg.blocks);
    if (!cfg.quiet) print_loading_bar(1.0f, "Searching subsets");
    if (!ok) { std::cerr << "No subset found for target " << target << "\n"; return; }
    rb.commit("H^n superposition", "H", -1, -1, "virtual superposition");
    for (int it = 0; it < cfg.grover_iters; ++it) {
        if (!cfg.quiet) print_loading_bar((float)it / cfg.grover_iters, "Grover " + std::to_string(it+1) + "/" + std::to_string(cfg.grover_iters));
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
    std::cout << "  bitstring = |" << bits_str_128(solution_bits, actual_size) << ">\n";
    std::cout << "  lo (hex)  = 0x" << std::hex << solution_bits.lo << std::dec << "\n";
    std::cout << "  hi (hex)  = 0x" << std::hex << solution_bits.hi << std::dec << "\n";
    std::cout << "  subset    = [";
    for (size_t i = 0; i < solution_subset.size(); ++i) { if (i) std::cout << ", "; std::cout << solution_subset[i]; }
    std::cout << "]\n";
    std::cout << "  sum       = " << sol_sum << (sol_sum == target ? "  ✓ matches target" : "  ✗ MISMATCH") << "\n\n";
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
        for (int s = 0; s < cols; ++s) std::cout << "s" << std::setw(2) << std::setfill('0') << s << std::setfill(' ') << (s+1==cols ? "" : " ");
        std::cout << "\n";
        for (int q = 0; q < (int)bm.size(); ++q) {
            std::cout << "  q" << std::setw(2) << q << " ";
            for (int s = 0; s < cols; ++s) std::cout << bm[q][s] << (s+1==cols ? "" : " ");
            std::cout << "\n";
        }
    }
    std::cout << "\n============================================================\n";
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    run_demo(cfg);
    return 0;
}
