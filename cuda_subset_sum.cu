#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(1); } } while(0)

// ---------------------------------------------------------------------------
// CLI config
// ---------------------------------------------------------------------------
struct Config {
    int  set_size     = 24;
    int  seed         = -1;       // -1 = time-based
    int  target       = -1;       // -1 = plant a random hidden subset
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
"GPU-accelerated subset-sum solver with virtual qubit ring buffer.\n"
"\n"
"Runtime options:\n"
"  --set-size <N>       Number of elements in the input set       [default: 24]\n"
"  --target <T>         Override target sum (skips random planting)\n"
"  --seed <S>           RNG seed for reproducible inputs          [default: time-based]\n"
"  --min-val <V>        Minimum element value                     [default: 1]\n"
"  --max-val <V>        Maximum element value                     [default: set-size*20]\n"
"\n"
"Solver options:\n"
"  --threads <N>        CUDA threads per block                    [default: 256]\n"
"  --blocks <N>         CUDA grid blocks                          [default: 256]\n"
"  --grover-iters <N>   Virtual Grover iteration count            [default: 1]\n"
"\n"
"Ring buffer:\n"
"  --ring-size <N>      VirtualQubitRing capacity (power of 2)    [default: 32]\n"
"  --bitmask-cols <N>   Basis states shown in 2D preview          [default: 32]\n"
"\n"
"Output:\n"
"  --verbose            Print full ring buffer slot dump\n"
"  --no-bitmask         Suppress 2D bitmask output\n"
"  --quiet              Print only the solution subset\n"
"  --help, -h           Show this message and exit\n"
"\n"
"Constraints:\n"
"  set-size must be <= 64 (bitmask fits in unsigned long long)\n"
"  threads * blocks should cover 2^set-size for a complete search\n"
"  ring-size must be a power of 2\n"
"\n"
"Examples:\n"
"  " << prog << "\n"
"  " << prog << " --set-size 20 --seed 42\n"
"  " << prog << " --set-size 28 --target 1337 --threads 512 --blocks 512\n"
"  " << prog << " --quiet --no-bitmask\n";
}

static int parse_int(const char* flag, const char* val) {
    if (!val) { std::cerr << flag << " requires an integer argument\n"; std::exit(1); }
    char* end;
    long v = std::strtol(val, &end, 10);
    if (*end != '\0') { std::cerr << flag << ": invalid integer '" << val << "'\n"; std::exit(1); }
    return (int)v;
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* next = (i + 1 < argc) ? argv[i + 1] : nullptr;

        if (!strcmp(a, "--help") || !strcmp(a, "-h")) { print_help(argv[0]); std::exit(0); }
        else if (!strcmp(a, "--set-size"))     { cfg.set_size     = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--target"))       { cfg.target       = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--seed"))         { cfg.seed         = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--min-val"))      { cfg.min_val      = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--max-val"))      { cfg.max_val      = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--threads"))      { cfg.threads      = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--blocks"))       { cfg.blocks       = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--grover-iters")) { cfg.grover_iters = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--ring-size"))    { cfg.ring_size    = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--bitmask-cols")) { cfg.bitmask_cols = parse_int(a, next); ++i; }
        else if (!strcmp(a, "--verbose"))      { cfg.verbose      = true; }
        else if (!strcmp(a, "--no-bitmask"))   { cfg.no_bitmask   = true; }
        else if (!strcmp(a, "--quiet"))        { cfg.quiet        = true; }
        else { std::cerr << "Unknown option: " << a << "  (try --help)\n"; std::exit(1); }
    }

    // derived defaults
    if (cfg.max_val < 0) cfg.max_val = cfg.set_size * 20;

    // validation
    if (cfg.set_size < 1 || cfg.set_size > 64) {
        std::cerr << "--set-size must be between 1 and 64\n"; std::exit(1);
    }
    if (cfg.min_val >= cfg.max_val) {
        std::cerr << "--min-val must be less than --max-val\n"; std::exit(1);
    }
    if (cfg.threads < 1 || cfg.blocks < 1) {
        std::cerr << "--threads and --blocks must be >= 1\n"; std::exit(1);
    }
    if (cfg.ring_size < 2 || (cfg.ring_size & (cfg.ring_size - 1)) != 0) {
        std::cerr << "--ring-size must be a power of 2 >= 2\n"; std::exit(1);
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
static void print_loading_bar(float progress, const std::string& prefix = "Loading", int width = 40) {
    progress = std::max(0.0f, std::min(1.0f, progress));
    int filled = (int)(width * progress);
    std::string bar(filled, '\xe2'); // UTF-8 block char via escape
    bar = std::string(filled, '#') + std::string(width - filled, '.');
    std::cout << "\r" << prefix << ": [" << bar << "] "
              << std::setw(5) << std::fixed << std::setprecision(1)
              << (progress * 100.0f) << "%" << std::flush;
    if (progress >= 1.0f) std::cout << "\n";
}

static std::vector<int> subset_from_bits(const std::vector<int>& numbers, unsigned long long bits) {
    std::vector<int> out;
    for (size_t i = 0; i < numbers.size(); ++i) if ((bits >> i) & 1ULL) out.push_back(numbers[i]);
    return out;
}

static std::string bits_str(unsigned long long bits, int width) {
    std::string s(width, '0');
    for (int i = 0; i < width; ++i) if ((bits >> i) & 1ULL) s[width - 1 - i] = '1';
    return s;
}

// ---------------------------------------------------------------------------
// Virtual qubit ring
// ---------------------------------------------------------------------------
struct VirtualSlot {
    std::string label;
    int head = 0, tail = 0;
    std::string gate;
    int target = -1, ctrl = -1;
    unsigned long long active_bits = 0;
    std::string notes;
};

class VirtualQubitRing {
public:
    int n, size, head, tail;
    unsigned long long active_bits;
    std::vector<VirtualSlot> slots;
    std::vector<std::tuple<std::string,int,int>> circuit_ops;

    VirtualQubitRing(int num_qubits, int ring_size = 32)
        : n(num_qubits), size(ring_size), head(0), tail(0), active_bits(0), slots(ring_size) {}

    void commit(const std::string& label, const std::string& gate = "",
                int target = -1, int ctrl = -1, const std::string& notes = "") {
        int nxt = (head + 1) % size;
        if (nxt == tail) tail = (tail + 1) % size;
        head = nxt;
        slots[head] = { label, head, tail, gate, target, ctrl, active_bits, notes };
        if (!gate.empty()) circuit_ops.push_back({gate, target, ctrl});
    }

    void set_active_bits(unsigned long long bits) { active_bits = bits; }

    std::vector<std::string> debug_summary() const {
        std::vector<std::string> lines;
        for (int i = 0; i < size; ++i) {
            const auto& s = slots[i];
            std::ostringstream oss;
            oss << "slot " << std::setw(2) << i << ": " << s.label;
            if (s.active_bits) oss << "  active_bits=0x" << std::hex << s.active_bits << std::dec;
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

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------
__global__ void subset_sum_kernel(const int* numbers, int n, int target,
                                   unsigned long long* out_solution, int* out_found) {
    unsigned long long idx    = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x  * blockDim.x;
    unsigned long long total  = 1ULL << n;
    for (unsigned long long mask = idx; mask < total; mask += stride) {
        int sum = 0;
        for (int j = 0; j < n; ++j) if ((mask >> j) & 1ULL) sum += numbers[j];
        if (sum == target) {
            if (atomicCAS(out_found, 0, 1) == 0) *out_solution = mask;
            return;
        }
    }
}

static bool gpu_subset_sum(const std::vector<int>& numbers, int target,
                            unsigned long long& solution, int threads, int blocks) {
    int n = (int)numbers.size();
    int *d_numbers = nullptr, *d_found = nullptr;
    unsigned long long *d_solution = nullptr;
    CUDA_CHECK(cudaMalloc(&d_numbers,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found,    sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_solution, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_numbers, numbers.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_solution, 0,  sizeof(unsigned long long)));

    subset_sum_kernel<<<blocks, threads>>>(d_numbers, n, target, d_solution, d_found);
    CUDA_CHECK(cudaDeviceSynchronize());

    int found = 0;
    CUDA_CHECK(cudaMemcpy(&found,    d_found,    sizeof(int),               cudaMemcpyDeviceToHost));
    if (found)
        CUDA_CHECK(cudaMemcpy(&solution, d_solution, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    cudaFree(d_numbers); cudaFree(d_found); cudaFree(d_solution);
    return found != 0;
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------
static void run_demo(const Config& cfg) {
    unsigned seed = (cfg.seed >= 0) ? (unsigned)cfg.seed
                                    : (unsigned)std::chrono::high_resolution_clock::now()
                                                   .time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(cfg.min_val, cfg.max_val);

    // build unique set
    std::vector<int> numbers;
    int attempts = 0;
    while ((int)numbers.size() < cfg.set_size) {
        if (++attempts > cfg.set_size * 1000) {
            std::cerr << "Could not generate " << cfg.set_size
                      << " unique values in [" << cfg.min_val << ", " << cfg.max_val << "]\n";
            std::exit(1);
        }
        int x = dist(rng);
        if (std::find(numbers.begin(), numbers.end(), x) == numbers.end())
            numbers.push_back(x);
    }
    std::sort(numbers.begin(), numbers.end());

    // determine target
    int target = cfg.target;
    std::vector<int> hidden_subset;
    if (target < 0) {
        std::uniform_int_distribution<int> ssd(5, std::min(cfg.set_size - 1, 6));
        int num_to_sum = ssd(rng);
        std::vector<int> idx(cfg.set_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int i = 0; i < num_to_sum; ++i) hidden_subset.push_back(numbers[idx[i]]);
        target = std::accumulate(hidden_subset.begin(), hidden_subset.end(), 0);
    }

    if (!cfg.quiet) {
        std::cout << "\n============================================================\n";
        std::cout << "Virtual-Qubit Subset-Sum Demo (CUDA/NVCC)\n";
        std::cout << "============================================================\n\n";
        std::cout << "Config: set-size=" << cfg.set_size
                  << "  seed=" << seed
                  << "  threads=" << cfg.threads
                  << "  blocks=" << cfg.blocks << "\n\n";
        std::cout << "INPUT SET:\n  [";
        for (size_t i = 0; i < numbers.size(); ++i) { if (i) std::cout << ", "; std::cout << numbers[i]; }
        std::cout << "]\n  size = " << numbers.size() << "\n\n";
        std::cout << "TARGET SUM: " << target << "\n\n";
        if (!hidden_subset.empty()) {
            std::cout << "PLANTED SUBSET (verification):\n  [";
            for (size_t i = 0; i < hidden_subset.size(); ++i) { if (i) std::cout << ", "; std::cout << hidden_subset[i]; }
            std::cout << "]\n\n";
        }
    }

    VirtualQubitRing rb(cfg.set_size, cfg.ring_size);
    rb.commit("|0...0> init");

    if (!cfg.quiet) print_loading_bar(0.0f, "Searching subsets");
    unsigned long long solution_bits = 0;
    bool ok = gpu_subset_sum(numbers, target, solution_bits, cfg.threads, cfg.blocks);
    if (!cfg.quiet) print_loading_bar(1.0f, "Searching subsets");

    if (!ok) { std::cerr << "No subset matches target " << target << "\n"; return; }

    rb.commit("H^n superposition", "H", -1, -1, "virtual superposition");
    for (int it = 0; it < cfg.grover_iters; ++it) {
        if (!cfg.quiet)
            print_loading_bar((float)it / cfg.grover_iters,
                              "Grover iter " + std::to_string(it+1) + "/" + std::to_string(cfg.grover_iters));
        rb.set_active_bits(solution_bits);
        rb.commit("Grover iter " + std::to_string(it+1), "G",
                  (int)solution_bits, -1, "virtual amplitude boost");
    }
    if (!cfg.quiet) print_loading_bar(1.0f, "Grover iterations");

    rb.set_active_bits(solution_bits);
    rb.commit("measurement", "M", (int)solution_bits, -1, "virtual measurement");

    auto solution_subset = subset_from_bits(numbers, solution_bits);
    if (cfg.quiet) {
        std::cout << "[";
        for (size_t i = 0; i < solution_subset.size(); ++i) { if (i) std::cout << ", "; std::cout << solution_subset[i]; }
        std::cout << "]\n";
        return;
    }

    std::cout << "\nSOLUTION:\n";
    std::cout << "  bitstring = |" << bits_str(solution_bits, cfg.set_size) << ">\n";
    std::cout << "  subset    = [";
    for (size_t i = 0; i < solution_subset.size(); ++i) { if (i) std::cout << ", "; std::cout << solution_subset[i]; }
    std::cout << "]\n";
    std::cout << "  sum       = " << std::accumulate(solution_subset.begin(), solution_subset.end(), 0) << "\n\n";

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
                      << (s + 1 == cols ? "" : " ");
        std::cout << "\n";
        for (int q = 0; q < (int)bm.size(); ++q) {
            std::cout << "  q" << std::setw(2) << q << " ";
            for (int s = 0; s < cols; ++s)
                std::cout << bm[q][s] << (s + 1 == cols ? "" : " ");
            std::cout << "\n";
        }
    }

    std::cout << "\n============================================================\n";
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    run_demo(cfg);
    return 0;
}
