// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/extension.hpp>
#include <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
public:
    FrontEnd();

    static std::shared_ptr<Model> convert(const InputModel::Ptr & model, bool naive = false);

    // Registers an OpenVINO extension with this frontend. A ConversionExtension contributes (or
    // overrides) the translator for its op type; anything else is ignored. The op then shows up in
    // get_supported_ops() and therefore in the backend's supports_op.
    static void add_extension(const std::shared_ptr<ov::Extension> & extension);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
