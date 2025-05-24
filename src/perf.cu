#include <string>
#include <unordered_map>
#include "perf.cuh"

#if PERF_ENABLED

// 全局统一性能数据存储的实现
std::unordered_map<std::string, PerfData>& get_perf_data() {
    static std::unordered_map<std::string, PerfData> g_perf_data;
    return g_perf_data;
}

#else

// 当性能测量被禁用时的空实现
std::unordered_map<std::string, PerfData>& get_perf_data() {
    static std::unordered_map<std::string, PerfData> empty_data;
    return empty_data;
}

#endif // PERF_ENABLED 