#pragma once

#include <memory>
#include <openvino/op/op.hpp>
#include <openvino/runtime/intel_gpu/external_op.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML_OP_RMS_NORM without weight: y = x / sqrt(mean(x^2) + eps) over the last dim.
// The GPU plugin does not lower it; a Triton kernel runs it on the plugin queue (see external_op.hpp).
class TritonRmsNorm : public ov::op::Op {
public:
    OPENVINO_OP("TritonRmsNorm", "ggml_openvino");

    TritonRmsNorm() = default;
    TritonRmsNorm(const ov::Output<ov::Node> & x, float eps);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector & new_args) const override;
    bool visit_attributes(ov::AttributeVisitor & visitor) override;

private:
    float m_eps = 0.0f;
};

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov

// Launcher for one RMS_NORM node, or nullptr when GGML_OPENVINO_TRITON_DIR is not set.
std::shared_ptr<ov::intel_gpu::ExternalOpCallback> ggml_openvino_triton_rms_norm(float eps);
