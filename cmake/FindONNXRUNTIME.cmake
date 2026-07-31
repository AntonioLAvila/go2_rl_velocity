# Locate an ONNX Runtime distribution.
#
# ONNX Runtime ships no ament package and no usable CMake config in the prebuilt
# tarballs, so we look for the header and library directly.
#
# Search order:
#   1. -DONNXRUNTIME_ROOT=<dir>   (also honoured natively via CMP0144)
#   2. $ENV{ONNXRUNTIME_ROOT}
#   3. system paths (/usr, /usr/local)
#
# On the Jetson, point ONNXRUNTIME_ROOT at your JetPack build:
#   colcon build --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
#
# Defines the imported target onnxruntime::onnxruntime.

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an ONNX Runtime distribution")

set(_ort_hints "")
if(ONNXRUNTIME_ROOT)
  list(APPEND _ort_hints "${ONNXRUNTIME_ROOT}")
endif()
if(DEFINED ENV{ONNXRUNTIME_ROOT})
  list(APPEND _ort_hints "$ENV{ONNXRUNTIME_ROOT}")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${_ort_hints}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
)

# The prebuilt tarballs ship only versioned sonames (libonnxruntime.so.1.22.0) with no
# unversioned symlink, so match those too rather than relying on the plain name.
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime libonnxruntime.so.1 libonnxruntime.so.1.22.0 libonnxruntime.so.1.23.0
  HINTS ${_ort_hints}
  PATH_SUFFIXES lib lib64
)

# ORT_API_VERSION, so a too-old runtime is caught at configure time rather than at model
# load. The exported policy is opset 18, which needs ONNX Runtime 1.15 or newer.
if(ONNXRUNTIME_INCLUDE_DIR AND EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
  file(STRINGS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" _ort_version_line
    REGEX "^#define ORT_API_VERSION[ \t]+[0-9]+")
  string(REGEX MATCH "[0-9]+" ONNXRUNTIME_VERSION "${_ort_version_line}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRUNTIME
  REQUIRED_VARS ONNXRUNTIME_LIBRARY ONNXRUNTIME_INCLUDE_DIR
  VERSION_VAR ONNXRUNTIME_VERSION
)

if(ONNXRUNTIME_FOUND AND ONNXRUNTIME_VERSION AND ONNXRUNTIME_VERSION LESS 15)
  message(FATAL_ERROR
    "ONNX Runtime API version ${ONNXRUNTIME_VERSION} is too old; the exported policy is "
    "opset 18 and needs ONNX Runtime 1.15 or newer.")
endif()

if(ONNXRUNTIME_FOUND AND NOT TARGET onnxruntime::onnxruntime)
  add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
