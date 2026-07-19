#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <nvperf_host.h>

#include <Eval.h>
#include <Metric.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define SET_ERROR(fmt, ...)                                                     \
    do {                                                                        \
        char buffer[1024];                                                      \
        std::snprintf(buffer, sizeof(buffer), fmt, ##__VA_ARGS__);              \
        gLastError = buffer;                                                    \
    } while (0)

#define DRIVER_CALL(call)                                                       \
    do {                                                                        \
        CUresult status__ = (call);                                             \
        if (status__ != CUDA_SUCCESS) {                                         \
            const char* err__ = nullptr;                                        \
            cuGetErrorString(status__, &err__);                                 \
            SET_ERROR("CUDA driver error: %s at %s:%d",                        \
                      err__ ? err__ : "unknown", __FILE__, __LINE__);          \
            return 0;                                                           \
        }                                                                       \
    } while (0)

#define CUPTI_CALL(call)                                                        \
    do {                                                                        \
        CUptiResult status__ = (call);                                          \
        if (status__ != CUPTI_SUCCESS) {                                        \
            const char* err__ = nullptr;                                        \
            cuptiGetResultString(status__, &err__);                             \
            SET_ERROR("CUPTI error: %s at %s:%d",                              \
                      err__ ? err__ : "unknown", __FILE__, __LINE__);          \
            return 0;                                                           \
        }                                                                       \
    } while (0)

std::string gLastError;
bool gInitialized = false;
bool gSessionActive = false;
CUcontext gContext = nullptr;
CUdevice gDevice = 0;
int gDeviceIndex = 0;
std::string gChipName;
std::string gCsvPath;
std::string gJsonPath;
std::vector<std::string> gMetricNames;
std::vector<uint8_t> gConfigImage;
std::vector<uint8_t> gCounterDataImage;
std::vector<uint8_t> gCounterDataScratchBuffer;
std::vector<uint8_t> gCounterDataImagePrefix;
std::vector<uint8_t> gCounterAvailabilityImage;
int gPassIndex = 0;

void progress(const char* message) {
    std::fprintf(stderr, "[msda_nvpw] %s\n", message);
    std::fflush(stderr);
}

std::vector<std::string> splitMetrics(const char* metricsCsv) {
    std::vector<std::string> metrics;
    std::stringstream ss(metricsCsv ? metricsCsv : "");
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) {
            metrics.push_back(item);
        }
    }
    return metrics;
}

bool createCounterDataImage(int maxRanges) {
    CUpti_Profiler_CounterDataImageOptions options;
    std::memset(&options, 0, sizeof(options));
    options.pCounterDataPrefix = gCounterDataImagePrefix.data();
    options.counterDataPrefixSize = gCounterDataImagePrefix.size();
    options.maxNumRanges = maxRanges;
    options.maxNumRangeTreeNodes = maxRanges;
    options.maxRangeNameLength = 128;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calc = {
        CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
    calc.pOptions = &options;
    calc.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_CALL(cuptiProfilerCounterDataImageCalculateSize(&calc));

    gCounterDataImage.assign(calc.counterDataImageSize, 0);
    CUpti_Profiler_CounterDataImage_Initialize_Params init = {
        CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    init.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    init.pOptions = &options;
    init.counterDataImageSize = gCounterDataImage.size();
    init.pCounterDataImage = gCounterDataImage.data();
    CUPTI_CALL(cuptiProfilerCounterDataImageInitialize(&init));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchSize = {
        CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchSize.counterDataImageSize = gCounterDataImage.size();
    scratchSize.pCounterDataImage = gCounterDataImage.data();
    CUPTI_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchSize));

    gCounterDataScratchBuffer.assign(scratchSize.counterDataScratchBufferSize, 0);
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params scratchInit = {
        CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    scratchInit.counterDataImageSize = gCounterDataImage.size();
    scratchInit.pCounterDataImage = gCounterDataImage.data();
    scratchInit.counterDataScratchBufferSize = gCounterDataScratchBuffer.size();
    scratchInit.pCounterDataScratchBuffer = gCounterDataScratchBuffer.data();
    CUPTI_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&scratchInit));
    return true;
}

bool writeCsvAndJson() {
    std::vector<NV::Metric::Eval::MetricNameValue> values;
    if (!NV::Metric::Eval::GetMetricGpuValue(gChipName, gCounterDataImage, gMetricNames, values, gCounterAvailabilityImage.data())) {
        SET_ERROR("NVPW metric evaluation failed");
        return false;
    }

    std::ofstream csv(gCsvPath);
    if (!csv) {
        SET_ERROR("failed to open CSV output: %s", gCsvPath.c_str());
        return false;
    }
    std::ofstream json(gJsonPath);
    if (!json) {
        SET_ERROR("failed to open JSON output: %s", gJsonPath.c_str());
        return false;
    }

    csv << "request_id,stage_name,range_name,metric_name,metric_value,metric_unit,kernel_name,shape_config,metric_source\n";
    json << "[\n";
    bool firstJson = true;
    for (const auto& metric : values) {
        for (const auto& rangeValue : metric.rangeNameMetricValueMap) {
            const std::string& rangeName = rangeValue.first;
            const double metricValue = rangeValue.second;
            const std::string unit =
                metric.metricName.find("pct") != std::string::npos ? "pct" :
                metric.metricName.find("per_cycle") != std::string::npos ? "inst/cycle" : "value";
            csv << "0,ms_deformable_attention," << rangeName << ','
                << metric.metricName << ',' << metricValue << ','
                << unit << ",\"msda_operator_range\",";
            csv << "\"captured_real_msda_replay\",cupti_range_nvpw_msda_replay\n";

            if (!firstJson) {
                json << ",\n";
            }
            firstJson = false;
            json << "  {"
                 << "\"request_id\":0,"
                 << "\"stage_name\":\"ms_deformable_attention\","
                 << "\"range_name\":\"" << rangeName << "\","
                 << "\"metric_name\":\"" << metric.metricName << "\","
                 << "\"metric_value\":" << metricValue << ','
                 << "\"metric_unit\":\"" << unit << "\","
                 << "\"kernel_name\":\"msda_operator_range\","
                 << "\"shape_config\":\"captured_real_msda_replay\","
                 << "\"metric_source\":\"cupti_range_nvpw_msda_replay\""
                 << "}";
        }
    }
    json << "\n]\n";
    return true;
}

}  // namespace

extern "C" const char* bev_msda_nvpw_last_error() {
    return gLastError.c_str();
}

extern "C" int bev_msda_nvpw_init(const char* metricsCsv, const char* csvPath, const char* jsonPath, int maxRanges) {
    progress("initializing CUDA/CUPTI/NVPW");
    gLastError.clear();
    gMetricNames = splitMetrics(metricsCsv);
    if (gMetricNames.empty()) {
        SET_ERROR("no metrics provided");
        return 0;
    }
    gCsvPath = csvPath && csvPath[0] ? csvPath : "reports/range_metrics_msda_nvpw.csv";
    gJsonPath = jsonPath && jsonPath[0] ? jsonPath : "reports/range_metrics_msda_nvpw.json";

    DRIVER_CALL(cuInit(0));
    DRIVER_CALL(cuDeviceGet(&gDevice, gDeviceIndex));
    CUresult currentStatus = cuCtxGetCurrent(&gContext);
    if (currentStatus != CUDA_SUCCESS || gContext == nullptr) {
        DRIVER_CALL(cuCtxCreate(&gContext, (CUctxCreateParams*)0, 0, gDevice));
    }

    CUpti_Profiler_Initialize_Params profilerInit = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerInitialize(&profilerInit));
    progress("CUPTI profiler initialized");

    CUpti_Profiler_DeviceSupported_Params supported = {CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
    supported.cuDevice = gDeviceIndex;
    supported.api = CUPTI_PROFILER_RANGE_PROFILING;
    CUPTI_CALL(cuptiProfilerDeviceSupported(&supported));
    if (supported.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
        SET_ERROR("CUPTI range profiling is not supported on this device/configuration");
        return 0;
    }

    CUpti_Device_GetChipName_Params chip = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chip.deviceIndex = gDeviceIndex;
    CUPTI_CALL(cuptiDeviceGetChipName(&chip));
    gChipName = chip.pChipName;

    CUpti_Profiler_GetCounterAvailability_Params avail = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    avail.ctx = gContext;
    CUPTI_CALL(cuptiProfilerGetCounterAvailability(&avail));
    gCounterAvailabilityImage.assign(avail.counterAvailabilityImageSize, 0);
    avail.pCounterAvailabilityImage = gCounterAvailabilityImage.data();
    CUPTI_CALL(cuptiProfilerGetCounterAvailability(&avail));

    NVPW_InitializeHost_Params nvpwInit = {NVPW_InitializeHost_Params_STRUCT_SIZE};
    if (NVPW_InitializeHost(&nvpwInit) != NVPA_STATUS_SUCCESS) {
        SET_ERROR("NVPW_InitializeHost failed");
        return 0;
    }
    progress("NVPW host initialized; building metric config images");
    if (!NV::Metric::Config::GetConfigImage(gChipName, gMetricNames, gConfigImage, gCounterAvailabilityImage.data())) {
        SET_ERROR("failed to create NVPW config image");
        return 0;
    }
    if (!NV::Metric::Config::GetCounterDataPrefixImage(gChipName, gMetricNames, gCounterDataImagePrefix, gCounterAvailabilityImage.data())) {
        SET_ERROR("failed to create NVPW counter data prefix image");
        return 0;
    }
    if (!createCounterDataImage(std::max(1, maxRanges))) {
        return 0;
    }
    progress("counter data image initialized");

    CUpti_Profiler_BeginSession_Params begin = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    begin.ctx = gContext;
    begin.counterDataImageSize = gCounterDataImage.size();
    begin.pCounterDataImage = gCounterDataImage.data();
    begin.counterDataScratchBufferSize = gCounterDataScratchBuffer.size();
    begin.pCounterDataScratchBuffer = gCounterDataScratchBuffer.data();
    begin.range = CUPTI_UserRange;
    begin.replayMode = CUPTI_UserReplay;
    begin.maxRangesPerPass = std::max(1, maxRanges);
    // MMCV's MSDA forward launches output-initialization work plus the core
    // deformable-im2col kernel, so range capacity must not equal range count.
    begin.maxLaunchesPerPass = 64;
    CUPTI_CALL(cuptiProfilerBeginSession(&begin));
    progress("profiling session started");

    CUpti_Profiler_SetConfig_Params config = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    config.pConfig = gConfigImage.data();
    config.configSize = gConfigImage.size();
    config.passIndex = 0;
    config.minNestingLevel = 1;
    config.numNestingLevels = 1;
    CUPTI_CALL(cuptiProfilerSetConfig(&config));
    progress("metric configuration installed");

    gInitialized = true;
    gSessionActive = true;
    gPassIndex = 0;
    return 1;
}

extern "C" int bev_msda_nvpw_begin_pass() {
    if (!gInitialized || !gSessionActive) {
        SET_ERROR("profiler is not initialized");
        return 0;
    }
    CUpti_Profiler_BeginPass_Params beginPass = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerBeginPass(&beginPass));
    CUpti_Profiler_EnableProfiling_Params enable = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerEnableProfiling(&enable));
    std::fprintf(stderr, "[msda_nvpw] collecting pass %d\n", gPassIndex + 1);
    std::fflush(stderr);
    return 1;
}

extern "C" int bev_msda_nvpw_push_range(const char* rangeName) {
    CUpti_Profiler_PushRange_Params push = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    push.pRangeName = rangeName && rangeName[0] ? rangeName : "range_ms_deformable_attention";
    CUPTI_CALL(cuptiProfilerPushRange(&push));
    return 1;
}

extern "C" int bev_msda_nvpw_pop_range() {
    CUpti_Profiler_PopRange_Params pop = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerPopRange(&pop));
    return 1;
}

extern "C" int bev_msda_nvpw_end_pass() {
    CUpti_Profiler_DisableProfiling_Params disable = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerDisableProfiling(&disable));
    CUpti_Profiler_EndPass_Params endPass = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerEndPass(&endPass));
    ++gPassIndex;
    std::fprintf(stderr, "[msda_nvpw] pass %d complete; all_passes=%d\n",
                 gPassIndex, endPass.allPassesSubmitted ? 1 : 0);
    std::fflush(stderr);
    return endPass.allPassesSubmitted ? 1 : 0;
}

extern "C" int bev_msda_nvpw_finalize() {
    if (!gSessionActive) {
        return 1;
    }
    CUpti_Profiler_FlushCounterData_Params flush = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerFlushCounterData(&flush));
    progress("counter data flushed; ending session");
    CUpti_Profiler_UnsetConfig_Params unset = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerUnsetConfig(&unset));
    CUpti_Profiler_EndSession_Params end = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerEndSession(&end));
    gSessionActive = false;
    CUpti_Profiler_DeInitialize_Params deinit = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerDeInitialize(&deinit));
    const bool wrote = writeCsvAndJson();
    progress(wrote ? "hardware metrics written" : "failed to write hardware metrics");
    return wrote ? 1 : 0;
}
