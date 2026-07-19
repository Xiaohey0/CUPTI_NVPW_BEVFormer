#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define CUPTI_CHECK(call)                                                       \
    do {                                                                        \
        CUptiResult status__ = (call);                                          \
        if (status__ != CUPTI_SUCCESS) {                                        \
            const char* errstr__ = nullptr;                                     \
            cuptiGetResultString(status__, &errstr__);                          \
            std::cerr << "CUPTI error: " << (errstr__ ? errstr__ : "unknown")  \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
        }                                                                       \
    } while (0)

namespace {

constexpr size_t kActivityBufferSize = 64 * 1024;

struct LaunchMetadata {
    uint64_t cpuEnterNs = 0;
    uint64_t cpuExitNs = 0;
    int requestId = -1;
    std::string stageName = "unattributed";
};

struct StageRange {
    int requestId = -1;
    std::string stageName;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
};

struct ActivityRecord {
    std::string activityType;
    std::string name;
    uint32_t correlationId = 0;
    uint32_t streamId = 0;
    uint64_t cpuEnterNs = 0;
    uint64_t cpuExitNs = 0;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
    uint64_t bytes = 0;
    int requestId = -1;
    std::string stageName = "unattributed";
};

std::mutex gMutex;
std::unordered_map<uint32_t, LaunchMetadata> gLaunches;
std::vector<ActivityRecord> gRecords;
std::vector<StageRange> gRanges;
std::vector<size_t> gStageStack;
std::atomic<int> gCurrentRequest{-1};
CUpti_SubscriberHandle gSubscriber = nullptr;
bool gInitialized = false;

uint64_t cuptiTimestamp() {
    uint64_t ts = 0;
    CUptiResult status = cuptiGetTimestamp(&ts);
    return status == CUPTI_SUCCESS ? ts : 0;
}

const char* envOrDefault(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && value[0] ? value : fallback;
}

std::string jsonEscape(const std::string& text) {
    std::ostringstream out;
    for (char c : text) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = kActivityBufferSize;
    *buffer = reinterpret_cast<uint8_t*>(std::malloc(*size));
    *maxNumRecords = 0;
}

void storeRecord(ActivityRecord record) {
    std::lock_guard<std::mutex> lock(gMutex);
    gRecords.push_back(std::move(record));
}

bool isTrackedRuntimeApi(CUpti_CallbackId cbid) {
    return cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020;
}

void handleKernel(const CUpti_Activity* raw) {
    if (raw->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
        raw->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        return;
    }
    const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel4*>(raw);
    ActivityRecord record;
    record.activityType = "kernel";
    record.name = kernel->name ? kernel->name : "unknown_kernel";
    record.correlationId = kernel->correlationId;
    record.streamId = kernel->streamId;
    record.startNs = kernel->start;
    record.endNs = kernel->end;
    storeRecord(std::move(record));
}

void handleMemcpy(const CUpti_Activity* raw) {
    if (raw->kind != CUPTI_ACTIVITY_KIND_MEMCPY) {
        return;
    }
    const auto* memcpy = reinterpret_cast<const CUpti_ActivityMemcpy*>(raw);
    ActivityRecord record;
    record.activityType = "memcpy";
    record.name = "cudaMemcpy";
    record.correlationId = memcpy->correlationId;
    record.streamId = memcpy->streamId;
    record.startNs = memcpy->start;
    record.endNs = memcpy->end;
    record.bytes = memcpy->bytes;
    storeRecord(std::move(record));
}

void handleMemset(const CUpti_Activity* raw) {
    if (raw->kind != CUPTI_ACTIVITY_KIND_MEMSET) {
        return;
    }
    const auto* memset = reinterpret_cast<const CUpti_ActivityMemset*>(raw);
    ActivityRecord record;
    record.activityType = "memset";
    record.name = "cudaMemset";
    record.correlationId = memset->correlationId;
    record.streamId = memset->streamId;
    record.startNs = memset->start;
    record.endNs = memset->end;
    record.bytes = memset->bytes;
    storeRecord(std::move(record));
}

void handleRuntimeApi(const CUpti_Activity* raw) {
    if (raw->kind != CUPTI_ACTIVITY_KIND_RUNTIME) {
        return;
    }
    const auto* api = reinterpret_cast<const CUpti_ActivityAPI*>(raw);
    if (!isTrackedRuntimeApi(api->cbid)) {
        return;
    }
    std::lock_guard<std::mutex> lock(gMutex);
    auto& launch = gLaunches[api->correlationId];
    launch.cpuEnterNs = api->start;
    launch.cpuExitNs = api->end;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
    CUpti_Activity* record = nullptr;
    while (true) {
        CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            handleRuntimeApi(record);
            handleKernel(record);
            handleMemcpy(record);
            handleMemset(record);
            continue;
        }
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        }
        break;
    }

    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    if (dropped > 0) {
        std::cerr << "[bevformer_activity_profiler] dropped CUPTI records: " << dropped << std::endl;
    }
    std::free(buffer);
}

void CUPTIAPI runtimeCallback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) {
    const auto* cbInfo = reinterpret_cast<const CUpti_CallbackData*>(cbdata);
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || cbInfo == nullptr) {
        return;
    }
    if (!isTrackedRuntimeApi(cbid)) {
        return;
    }

    std::lock_guard<std::mutex> lock(gMutex);
    auto& launch = gLaunches[cbInfo->correlationId];
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        launch.cpuEnterNs = cuptiTimestamp();
        launch.requestId = gCurrentRequest.load();
        if (!gStageStack.empty()) {
            const size_t idx = gStageStack.back();
            if (idx < gRanges.size()) {
                launch.stageName = gRanges[idx].stageName;
            }
        }
    } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        launch.cpuExitNs = cuptiTimestamp();
    }
}

void enableActivityKind(CUpti_ActivityKind kind) {
    CUptiResult status = cuptiActivityEnable(kind);
    if (status != CUPTI_SUCCESS && status != CUPTI_ERROR_NOT_COMPATIBLE) {
        const char* errstr = nullptr;
        cuptiGetResultString(status, &errstr);
        std::cerr << "[bevformer_activity_profiler] enable kind " << static_cast<int>(kind)
                  << " failed: " << (errstr ? errstr : "unknown") << std::endl;
    }
}

void assignStages(std::vector<ActivityRecord>& records, const std::vector<StageRange>& ranges) {
    for (auto& record : records) {
        if (record.stageName != "unattributed") {
            continue;
        }
        const uint64_t midpoint = record.startNs + (record.endNs > record.startNs ? (record.endNs - record.startNs) / 2 : 0);
        uint64_t bestDuration = UINT64_MAX;
        for (const auto& range : ranges) {
            if (range.startNs <= midpoint && midpoint <= range.endNs) {
                const uint64_t duration = range.endNs > range.startNs ? range.endNs - range.startNs : UINT64_MAX - 1;
                if (duration >= bestDuration) {
                    continue;
                }
                record.requestId = range.requestId;
                record.stageName = range.stageName;
                bestDuration = duration;
            }
        }
    }
}

void assignLaunchMetadata(
    std::vector<ActivityRecord>& records,
    const std::unordered_map<uint32_t, LaunchMetadata>& launches) {
    for (auto& record : records) {
        const auto launch = launches.find(record.correlationId);
        if (launch == launches.end()) {
            continue;
        }
        if (launch->second.cpuEnterNs && record.startNs < launch->second.cpuEnterNs) {
            continue;
        }
        record.cpuEnterNs = launch->second.cpuEnterNs;
        record.cpuExitNs = launch->second.cpuExitNs;
        record.requestId = launch->second.requestId;
        record.stageName = launch->second.stageName;
    }
}

uint64_t baseTimestamp(const std::vector<ActivityRecord>& records, const std::vector<StageRange>& ranges) {
    uint64_t base = UINT64_MAX;
    for (const auto& r : records) {
        if (r.cpuEnterNs) base = std::min(base, r.cpuEnterNs);
        if (r.startNs) base = std::min(base, r.startNs);
    }
    for (const auto& range : ranges) {
        if (range.startNs) base = std::min(base, range.startNs);
    }
    return base == UINT64_MAX ? 0 : base;
}

double toUs(uint64_t ns, uint64_t base) {
    if (ns == 0 || ns < base) {
        return 0.0;
    }
    return static_cast<double>(ns - base) / 1000.0;
}

void writeCsv(const std::vector<ActivityRecord>& records, uint64_t base) {
    const char* path = envOrDefault("BEV_ACTIVITY_CSV", "reports/activity_timeline.csv");
    std::ofstream out(path);
    if (!out) {
        std::cerr << "[bevformer_activity_profiler] failed to open CSV: " << path << std::endl;
        return;
    }

    out << std::setprecision(15);
    out << "request_id,stage_name,activity_type,kernel_name,correlation_id,stream_id,"
        << "cpu_launch_us,cpu_launch_end_us,gpu_start_us,gpu_end_us,gpu_duration_us,"
        << "cpu_launch_overhead_us,scheduling_delay_us,launch_to_start_us,memcpy_bytes\n";
    for (const auto& record : records) {
        const double cpuUs = toUs(record.cpuEnterNs, base);
        const double cpuEndUs = toUs(record.cpuExitNs, base);
        const double startUs = toUs(record.startNs, base);
        const double endUs = toUs(record.endNs, base);
        const double durationUs = record.endNs > record.startNs ? static_cast<double>(record.endNs - record.startNs) / 1000.0 : 0.0;
        const double launchOverheadUs = (record.cpuEnterNs && record.cpuExitNs > record.cpuEnterNs)
            ? static_cast<double>(record.cpuExitNs - record.cpuEnterNs) / 1000.0
            : 0.0;
        const double schedulingDelayUs = (record.cpuExitNs && record.startNs > record.cpuExitNs)
            ? static_cast<double>(record.startNs - record.cpuExitNs) / 1000.0
            : 0.0;
        const double launchDelayUs = (record.cpuEnterNs && record.startNs > record.cpuEnterNs)
            ? static_cast<double>(record.startNs - record.cpuEnterNs) / 1000.0
            : 0.0;
        out << record.requestId << ','
            << record.stageName << ','
            << record.activityType << ','
            << '"' << record.name << '"' << ','
            << record.correlationId << ','
            << record.streamId << ','
            << cpuUs << ','
            << cpuEndUs << ','
            << startUs << ','
            << endUs << ','
            << durationUs << ','
            << launchOverheadUs << ','
            << schedulingDelayUs << ','
            << launchDelayUs << ','
            << record.bytes << '\n';
    }
}

void writeJson(const std::vector<ActivityRecord>& records, uint64_t base) {
    const char* path = envOrDefault("BEV_ACTIVITY_JSON", "reports/activity_timeline.json");
    std::ofstream out(path);
    if (!out) {
        std::cerr << "[bevformer_activity_profiler] failed to open JSON: " << path << std::endl;
        return;
    }
    out << std::setprecision(15);
    out << "[\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        const double durationUs = record.endNs > record.startNs ? static_cast<double>(record.endNs - record.startNs) / 1000.0 : 0.0;
        const double launchOverheadUs = (record.cpuEnterNs && record.cpuExitNs > record.cpuEnterNs)
            ? static_cast<double>(record.cpuExitNs - record.cpuEnterNs) / 1000.0
            : 0.0;
        const double schedulingDelayUs = (record.cpuExitNs && record.startNs > record.cpuExitNs)
            ? static_cast<double>(record.startNs - record.cpuExitNs) / 1000.0
            : 0.0;
        const double launchDelayUs = (record.cpuEnterNs && record.startNs > record.cpuEnterNs)
            ? static_cast<double>(record.startNs - record.cpuEnterNs) / 1000.0
            : 0.0;
        out << "  {"
            << "\"request_id\":" << record.requestId << ','
            << "\"stage_name\":\"" << jsonEscape(record.stageName) << "\","
            << "\"activity_type\":\"" << jsonEscape(record.activityType) << "\","
            << "\"kernel_name\":\"" << jsonEscape(record.name) << "\","
            << "\"correlation_id\":" << record.correlationId << ','
            << "\"stream_id\":" << record.streamId << ','
            << "\"cpu_launch_us\":" << toUs(record.cpuEnterNs, base) << ','
            << "\"cpu_launch_end_us\":" << toUs(record.cpuExitNs, base) << ','
            << "\"gpu_start_us\":" << toUs(record.startNs, base) << ','
            << "\"gpu_end_us\":" << toUs(record.endNs, base) << ','
            << "\"gpu_duration_us\":" << durationUs << ','
            << "\"cpu_launch_overhead_us\":" << launchOverheadUs << ','
            << "\"scheduling_delay_us\":" << schedulingDelayUs << ','
            << "\"launch_to_start_us\":" << launchDelayUs << ','
            << "\"memcpy_bytes\":" << record.bytes
            << "}";
        if (i + 1 != records.size()) out << ',';
        out << '\n';
    }
    out << "]\n";
}

void initializeProfiler() {
    if (gInitialized) {
        return;
    }
    gInitialized = true;

    cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    enableActivityKind(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    enableActivityKind(CUPTI_ACTIVITY_KIND_KERNEL);
    enableActivityKind(CUPTI_ACTIVITY_KIND_RUNTIME);
    enableActivityKind(CUPTI_ACTIVITY_KIND_MEMCPY);
    enableActivityKind(CUPTI_ACTIVITY_KIND_MEMSET);

    CUPTI_CHECK(cuptiSubscribe(&gSubscriber, runtimeCallback, nullptr));
    if (gSubscriber) {
        CUPTI_CHECK(cuptiEnableDomain(1, gSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    }
}

void finalizeProfiler() {
    if (!gInitialized) {
        return;
    }
    cudaDeviceSynchronize();
    cuptiActivityFlushAll(0);

    std::vector<ActivityRecord> records;
    std::vector<StageRange> ranges;
    std::unordered_map<uint32_t, LaunchMetadata> launches;
    {
        std::lock_guard<std::mutex> lock(gMutex);
        records = gRecords;
        ranges = gRanges;
        launches = gLaunches;
    }

    std::sort(records.begin(), records.end(), [](const ActivityRecord& lhs, const ActivityRecord& rhs) {
        if (lhs.startNs != rhs.startNs) return lhs.startNs < rhs.startNs;
        return lhs.correlationId < rhs.correlationId;
    });
    assignLaunchMetadata(records, launches);
    assignStages(records, ranges);
    const uint64_t base = baseTimestamp(records, ranges);
    writeCsv(records, base);
    writeJson(records, base);

    if (gSubscriber) {
        cuptiUnsubscribe(gSubscriber);
        gSubscriber = nullptr;
    }
}

}  // namespace

extern "C" void bev_profiler_begin_request(int request_id) {
    initializeProfiler();
    gCurrentRequest.store(request_id);
}

extern "C" void bev_profiler_end_request() {
    gCurrentRequest.store(-1);
}

extern "C" void bev_profiler_push_stage(const char* stage_name) {
    initializeProfiler();
    StageRange range;
    range.requestId = gCurrentRequest.load();
    range.stageName = stage_name && stage_name[0] ? stage_name : "unknown_stage";
    range.startNs = cuptiTimestamp();

    std::lock_guard<std::mutex> lock(gMutex);
    gRanges.push_back(std::move(range));
    gStageStack.push_back(gRanges.size() - 1);
}

extern "C" void bev_profiler_pop_stage() {
    const uint64_t endNs = cuptiTimestamp();
    std::lock_guard<std::mutex> lock(gMutex);
    if (gStageStack.empty()) {
        return;
    }
    const size_t idx = gStageStack.back();
    gStageStack.pop_back();
    if (idx < gRanges.size()) {
        gRanges[idx].endNs = endNs;
    }
}

__attribute__((constructor)) static void bevActivityConstructor() {
    initializeProfiler();
}

__attribute__((destructor)) static void bevActivityDestructor() {
    finalizeProfiler();
}
