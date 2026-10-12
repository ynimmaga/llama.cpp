#include "triton_rms_norm.h"

#include "../ggml-openvino-extra.h"
#include "ggml-impl.h"

#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <ze_api.h>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

TritonRmsNorm::TritonRmsNorm(const ov::Output<ov::Node> & x, float eps) : ov::op::Op({x}), m_eps(eps) {
    constructor_validate_and_infer_types();
}

void TritonRmsNorm::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> TritonRmsNorm::clone_with_new_inputs(const ov::OutputVector & new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TritonRmsNorm>(new_args.at(0), m_eps);
}

bool TritonRmsNorm::visit_attributes(ov::AttributeVisitor & visitor) {
    visitor.on_attribute("eps", m_eps);
    return true;
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

namespace {

#define ZE_CHECK(call)                                                       \
    do {                                                                     \
        ze_result_t r_ = (call);                                             \
        if (r_ != ZE_RESULT_SUCCESS) {                                       \
            GGML_ABORT("ggml-openvino triton: %s failed: 0x%x", #call, r_); \
        }                                                                    \
    } while (0)

// One AOT-compiled Triton variant (see triton/aot_compile.py) and its launch metadata.
struct triton_variant {
    std::string x_dtype;
    std::string y_dtype;
    int64_t block;
    std::string name;
    std::string build_flags;
    std::vector<uint8_t> spirv;
    uint32_t group_size;  // num_warps * threads_per_warp
    uint32_t shared;      // local memory bytes, passed as the last kernel argument when non-zero
};

// Loads the variants once and builds a Level Zero kernel per (context, variant) on first use.
// Kernel arguments are state on the kernel object, so set + append happen under one lock.
class triton_rms_norm_kernels {
public:
    explicit triton_rms_norm_kernels(const std::string & dir) {
        std::ifstream mf(dir + "/manifest.json");
        if (!mf) {
            GGML_ABORT("ggml-openvino triton: cannot read %s/manifest.json", dir.c_str());
        }
        auto manifest = nlohmann::json::parse(mf);
        for (const auto & v : manifest["variants"]) {
            triton_variant tv;
            tv.x_dtype = v["x_dtype"];
            tv.y_dtype = v["y_dtype"];
            tv.block = v["block"];
            tv.name = v["name"];
            tv.build_flags = v["build_flags"];
            tv.group_size = v["num_warps"].get<uint32_t>() * v["threads_per_warp"].get<uint32_t>();
            tv.shared = v["shared"];
            if (v["global_scratch_size"].get<int64_t>() != 0 || v["profile_scratch_size"].get<int64_t>() != 0) {
                GGML_ABORT("ggml-openvino triton: variant %s needs scratch memory, not supported", tv.name.c_str());
            }
            std::ifstream sf(dir + "/" + v["file"].get<std::string>(), std::ios::binary);
            tv.spirv.assign(std::istreambuf_iterator<char>(sf), std::istreambuf_iterator<char>());
            if (tv.spirv.empty()) {
                GGML_ABORT("ggml-openvino triton: empty SPIR-V for %s", v["file"].get<std::string>().c_str());
            }
            variants.push_back(std::move(tv));
        }
        GGML_LOG_INFO("ggml-openvino: triton RMS_NORM enabled, %zu kernel variants from %s\n", variants.size(),
                      dir.c_str());
    }

    void launch(ze_command_list_handle_t cmd_list,
                const ov::intel_gpu::ExternalOpTensor & x,
                const ov::intel_gpu::ExternalOpTensor & y,
                float eps) {
        auto dtype_name = [](const ov::element::Type & t) -> const char * {
            return t == ov::element::f32 ? "f32" : t == ov::element::f16 ? "f16" : nullptr;
        };
        const char * x_dtype = dtype_name(x.type);
        const char * y_dtype = dtype_name(y.type);
        if (x_dtype == nullptr || y_dtype == nullptr || y.shape != x.shape) {
            GGML_ABORT("ggml-openvino triton: unsupported RMS_NORM tensors x=%s%s y=%s%s", x.type.get_type_name().c_str(),
                       x.shape.to_string().c_str(), y.type.get_type_name().c_str(), y.shape.to_string().c_str());
        }
        // As 4D [D0, D1, D2, n_cols]: outer dims and their strides, padded with leading size-1 dims.
        const size_t rank = x.shape.size();
        if (rank < 1 || rank > 4 || x.strides.size() != rank || y.strides.size() != rank || x.strides.back() != 1 ||
            y.strides.back() != 1) {
            GGML_ABORT("ggml-openvino triton: RMS_NORM needs rank 1-4 tensors with a contiguous last dim");
        }
        int32_t dims[3] = { 1, 1, 1 };
        int32_t sx[3] = { 0, 0, 0 };
        int32_t sy[3] = { 0, 0, 0 };
        for (size_t i = 0; i + 1 < rank; i++) {
            const size_t d = 3 - (rank - 1) + i;
            dims[d] = static_cast<int32_t>(x.shape[i]);
            sx[d] = static_cast<int32_t>(x.strides[i]);
            sy[d] = static_cast<int32_t>(y.strides[i]);
        }
        const int32_t n_cols = static_cast<int32_t>(x.shape.back());
        const uint32_t n_rows = static_cast<uint32_t>(dims[0]) * dims[1] * dims[2];
        if (n_rows == 0) {
            return;
        }

        const triton_variant * var = nullptr;
        for (const auto & v : variants) {
            if (v.x_dtype == x_dtype && v.y_dtype == y_dtype && v.block >= n_cols &&
                (var == nullptr || v.block < var->block)) {
                var = &v;
            }
        }
        if (var == nullptr) {
            GGML_ABORT("ggml-openvino triton: no %s->%s variant for a row of %d", x_dtype, y_dtype, n_cols);
        }

        std::lock_guard<std::mutex> lock(mtx);
        ze_kernel_handle_t kernel = get_kernel(cmd_list, *var);

        // Same argument order as Triton's XPU launcher: user args, global and profile scratch, then local memory.
        const int32_t ints[8] = { dims[1], dims[2], sx[0], sx[1], sx[2], sy[0], sy[1], sy[2] };
        void * no_scratch = nullptr;
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &x.data));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &y.data));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(n_cols), &n_cols));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(eps), &eps));
        for (uint32_t i = 0; i < 8; i++) {
            ZE_CHECK(zeKernelSetArgumentValue(kernel, 4 + i, sizeof(int32_t), &ints[i]));
        }
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 12, sizeof(void *), &no_scratch));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 13, sizeof(void *), &no_scratch));
        if (var->shared != 0) {
            ZE_CHECK(zeKernelSetArgumentValue(kernel, 14, var->shared, nullptr));
        }
        ZE_CHECK(zeKernelSetGroupSize(kernel, var->group_size, 1, 1));
        ze_group_count_t groups = { n_rows, 1, 1 };
        ZE_CHECK(zeCommandListAppendLaunchKernel(cmd_list, kernel, &groups, nullptr, 0, nullptr));
    }

private:
    ze_kernel_handle_t get_kernel(ze_command_list_handle_t cmd_list, const triton_variant & var) {
        ze_context_handle_t ctx = nullptr;
        ze_device_handle_t dev = nullptr;
        ZE_CHECK(zeCommandListGetContextHandle(cmd_list, &ctx));
        ZE_CHECK(zeCommandListGetDeviceHandle(cmd_list, &dev));
        auto key = std::make_pair(ctx, &var);
        auto it = kernels.find(key);
        if (it != kernels.end()) {
            return it->second;
        }

        ze_module_desc_t mdesc = {};
        mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
        mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        mdesc.inputSize = var.spirv.size();
        mdesc.pInputModule = var.spirv.data();
        mdesc.pBuildFlags = var.build_flags.c_str();
        ze_module_handle_t module = nullptr;
        ze_module_build_log_handle_t log = nullptr;
        ze_result_t r = zeModuleCreate(ctx, dev, &mdesc, &module, &log);
        if (r != ZE_RESULT_SUCCESS) {
            size_t n = 0;
            zeModuleBuildLogGetString(log, &n, nullptr);
            std::string text(n, '\0');
            zeModuleBuildLogGetString(log, &n, text.data());
            GGML_ABORT("ggml-openvino triton: zeModuleCreate(%s) failed 0x%x: %s", var.name.c_str(), r, text.c_str());
        }
        zeModuleBuildLogDestroy(log);

        ze_kernel_desc_t kdesc = {};
        kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        kdesc.pKernelName = var.name.c_str();
        ze_kernel_handle_t kernel = nullptr;
        ZE_CHECK(zeKernelCreate(module, &kdesc, &kernel));

        ze_kernel_properties_t props = {};
        props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
        ZE_CHECK(zeKernelGetProperties(kernel, &props));
        const uint32_t expected_args = 14 + (var.shared != 0 ? 1 : 0);
        if (props.numKernelArgs != expected_args) {
            GGML_ABORT("ggml-openvino triton: %s has %u args, expected %u", var.name.c_str(), props.numKernelArgs,
                       expected_args);
        }
        kernels[key] = kernel;
        return kernel;
    }

    std::vector<triton_variant> variants;
    std::map<std::pair<ze_context_handle_t, const triton_variant *>, ze_kernel_handle_t> kernels;
    std::mutex mtx;
};

// Per-node callback: eps differs between nodes, the kernels are shared.
class triton_rms_norm_launch : public ov::intel_gpu::ExternalOpCallback {
public:
    triton_rms_norm_launch(std::shared_ptr<triton_rms_norm_kernels> kernels, float eps) :
        kernels(std::move(kernels)),
        eps(eps) {}

    void enqueue(void * native_queue,
                 const std::vector<ov::intel_gpu::ExternalOpTensor> & inputs,
                 const std::vector<ov::intel_gpu::ExternalOpTensor> & outputs) override {
        GGML_ASSERT(inputs.size() == 1 && outputs.size() == 1);
        kernels->launch(static_cast<ze_command_list_handle_t>(native_queue), inputs[0], outputs[0], eps);
    }

private:
    std::shared_ptr<triton_rms_norm_kernels> kernels;
    float eps;
};

}  // namespace

std::shared_ptr<ov::intel_gpu::ExternalOpCallback> ggml_openvino_triton_rms_norm(float eps) {
    static std::shared_ptr<triton_rms_norm_kernels> kernels = []() -> std::shared_ptr<triton_rms_norm_kernels> {
        const char * dir = ggml_openvino_getenv_str("GGML_OPENVINO_TRITON_DIR");
        if (dir == nullptr || !ggml_openvino_is_gpu() || !ggml_openvino_gpu_is_level_zero()) {
            return nullptr;
        }
        return std::make_shared<triton_rms_norm_kernels>(dir);
    }();
    if (!kernels) {
        return nullptr;
    }
    return std::make_shared<triton_rms_norm_launch>(kernels, eps);
}
