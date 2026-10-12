find_package(OpenVINO REQUIRED COMPONENTS Runtime Threading)
find_package(OpenCL REQUIRED)

message(STATUS "Found OpenVINO: ${OpenVINO_DIR} (found version \"${OpenVINO_VERSION}\")")

file(GLOB_RECURSE GGML_HEADERS_OPENVINO "*.h" "*.hpp")
file(GLOB_RECURSE GGML_SOURCES_OPENVINO "*.cpp")

# Experimental: GGML_OP_RMS_NORM as a Triton kernel on the OpenVINO GPU plugin queue (Level Zero plugin build only)
if (NOT GGML_OPENVINO_TRITON)
    list(FILTER GGML_SOURCES_OPENVINO EXCLUDE REGEX "/triton/")
    list(FILTER GGML_HEADERS_OPENVINO EXCLUDE REGEX "/triton/")
endif()

ggml_add_backend_library(ggml-openvino
    ${GGML_SOURCES_OPENVINO}
    ${GGML_HEADERS_OPENVINO}
)

target_link_libraries(ggml-openvino PRIVATE openvino::runtime openvino::threading OpenCL::OpenCL)

if (GGML_OPENVINO_TRITON)
    find_path(LEVEL_ZERO_INCLUDE_DIR ze_api.h PATH_SUFFIXES level_zero REQUIRED)
    find_library(LEVEL_ZERO_LIBRARY NAMES ze_loader libze_loader.so.1 REQUIRED)
    target_include_directories(ggml-openvino PRIVATE ${LEVEL_ZERO_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/vendor)
    target_link_libraries(ggml-openvino PRIVATE ${LEVEL_ZERO_LIBRARY})
    target_compile_definitions(ggml-openvino PRIVATE GGML_OPENVINO_TRITON)
endif()

# the OpenVINO RTTI macros take one argument and leave __VA_ARGS__ empty, which -Wpedantic reports
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ggml-openvino PRIVATE -Wno-gnu-zero-variadic-macro-arguments)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ggml-openvino PRIVATE -Wno-pedantic)
endif()

if (WIN32)
    target_link_libraries(ggml-openvino PRIVATE psapi)
endif()

if (GGML_OPENVINO)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    else()
        message(FATAL_ERROR "OpenVINO: OpenVINO toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()
