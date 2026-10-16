#include "frontend.h"

#include "ggml-impl.h"
#include "ggml-openvino/ggml-openvino-extra.h"
#include "input_model.h"
#include "op_table.h"
#include "translate_session.h"
#include "utils.h"

#include <openvino/core/type.hpp>
#include <openvino/frontend/extension/conversion.hpp>

#include <cstdlib>
#include <mutex>
#include <sstream>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace ov {
namespace frontend {
namespace ggml {

namespace {

// What a ConversionExtension converter sees: the ggml node, but with get_input(i) returning the
// view-resolved input (as built-in translators get via process_view_input_new), and the raw
// op_params available as the "op_params" attribute.
using GgmlNodeContext = ov::frontend::ggml::NodeContext;

class ExtensionNodeContext : public ov::frontend::NodeContext {
public:
    explicit ExtensionNodeContext(const GgmlNodeContext & inner) :
        ov::frontend::NodeContext(inner.get_op_type()),
        m_inner(inner) {}

    size_t get_input_size() const override { return m_inner.get_input_size(); }

    Output<Node> get_input(int idx) const override { return process_view_input_new(m_inner, idx); }

    Output<Node> get_input(const std::string & name) const override { return m_inner.get_input(name); }

    const std::string & get_name() const override { return m_inner.get_name(); }

    ov::Any get_attribute_as_any(const std::string & name) const override {
        return m_inner.get_attribute_as_any(name);
    }

private:
    const GgmlNodeContext & m_inner;
};

#ifdef _WIN32
void * ext_dl_open(const std::string & path) { return reinterpret_cast<void *>(LoadLibraryA(path.c_str())); }
void * ext_dl_sym(void * handle, const char * name) {
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
}
constexpr char kPathSep = ';';
#else
void * ext_dl_open(const std::string & path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }
void * ext_dl_sym(void * handle, const char * name) { return dlsym(handle, name); }
constexpr char kPathSep = ':';
#endif

}  // namespace

FrontEnd::FrontEnd() {}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension> & extension) {
    auto conversion = ov::as_type_ptr<ov::frontend::ConversionExtension>(extension);
    if (!conversion) {
        return;
    }
    auto converter = conversion->get_converter();
    register_extension_op(conversion->get_op_type(), [converter](const NodeContext & context) -> OutputVector {
        ExtensionNodeContext wrapped(context);
        return converter(wrapped);
    });
}

// Loads every library listed in GGML_OPENVINO_EXTENSIONS (':'-separated, ';' on Windows). Each one
// must export the standard OpenVINO entry point (OPENVINO_CREATE_EXTENSIONS). Its extensions are
// added to the OpenVINO Core (core.add_extension) and to this frontend.
void ensure_extensions_loaded() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char * env = ggml_openvino_getenv_str("GGML_OPENVINO_EXTENSIONS");
        if (env == nullptr || *env == '\0') {
            return;
        }
        std::stringstream list(env);
        std::string path;
        while (std::getline(list, path, kPathSep)) {
            if (path.empty()) {
                continue;
            }
            void * handle = ext_dl_open(path);
            if (handle == nullptr) {
                GGML_LOG_ERROR("ggml-openvino: cannot load extension library %s\n", path.c_str());
                continue;
            }
            using CreateFn = void (*)(std::vector<std::shared_ptr<ov::Extension>> &);
            auto create = reinterpret_cast<CreateFn>(ext_dl_sym(handle, "create_extensions"));
            if (create == nullptr) {
                GGML_LOG_ERROR("ggml-openvino: %s has no create_extensions entry point\n", path.c_str());
                continue;
            }
            std::vector<std::shared_ptr<ov::Extension>> extensions;
            create(extensions);
            ov_singleton_core().add_extension(extensions);
            for (const auto & extension : extensions) {
                FrontEnd::add_extension(extension);
            }
            GGML_LOG_INFO("ggml-openvino: loaded %zu extension(s) from %s\n", extensions.size(), path.c_str());
        }
    });
}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr & model, bool naive) {
    auto ggml_model = ov::as_type_ptr<ggml::InputModel>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    const auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    return converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
