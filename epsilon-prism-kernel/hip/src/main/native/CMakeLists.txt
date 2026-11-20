# HIP ランタイムは DJL が先にロードする。学習で使う libtorch のシンボルは実行時に取得する。
cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1100)
endif()

project(prism_hip LANGUAGES CXX HIP)

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_SKIP_BUILD_RPATH ON)
set(CMAKE_SKIP_INSTALL_RPATH ON)

foreach(required PRISM_JAVA_HOME PRISM_OUTPUT_DIR PRISM_TORCH_VERSION PRISM_TORCH_FLAVOR)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} must be given (the Maven profile native-hip passes it)")
  endif()
endforeach()

list(LENGTH CMAKE_HIP_ARCHITECTURES PRISM_ARCH_COUNT)
if(NOT PRISM_ARCH_COUNT EQUAL 1)
  message(FATAL_ERROR "exactly one HIP architecture is supported, got ${CMAKE_HIP_ARCHITECTURES}")
endif()
set(PRISM_HIP_ARCH ${CMAKE_HIP_ARCHITECTURES})

set(PRISM_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common")
set(PRISM_INFERENCE_DIR "${PRISM_COMMON}/src/main/native/inference")
file(STRINGS "${PRISM_INFERENCE_DIR}/inference_sources.txt" PRISM_INFERENCE_SOURCES)
list(TRANSFORM PRISM_INFERENCE_SOURCES PREPEND "${PRISM_INFERENCE_DIR}/")
set(PRISM_KERNELS ${PRISM_INFERENCE_SOURCES})
set_source_files_properties(${PRISM_INFERENCE_SOURCES} PROPERTIES LANGUAGE HIP)
set(PRISM_TRAINING_DIR "${PRISM_COMMON}/src/main/native/training")
file(STRINGS "${PRISM_TRAINING_DIR}/training_sources.txt" PRISM_TRAINING_SOURCES)
list(TRANSFORM PRISM_TRAINING_SOURCES PREPEND "${PRISM_TRAINING_DIR}/")
set_source_files_properties(${PRISM_TRAINING_SOURCES} PROPERTIES LANGUAGE HIP)
set(PRISM_CACHE_SOURCES "${PRISM_INFERENCE_DIR}/../cache/candidate_offset_cache.cc")
set(PRISM_INFERENCE_OBJECT_SOURCES device_stream.cc ${PRISM_CACHE_SOURCES} ${PRISM_KERNELS})
set(PRISM_LIBRARY_SOURCES declaration_jni.cc prism_hip_jni.cc prism_hip_runtime.cc
  prism_hip_torch.cc ${PRISM_TRAINING_SOURCES})
set_source_files_properties(device_stream.cc prism_hip_jni.cc prism_hip_runtime.cc prism_hip_torch.cc ${PRISM_CACHE_SOURCES}
  PROPERTIES LANGUAGE HIP)

function(prism_hip_compile name checked)
  target_compile_features(${name} PRIVATE cxx_std_17)
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PRISM_INFERENCE_DIR}
    "${PRISM_INFERENCE_DIR}/.."
    ${PRISM_JAVA_HOME}/include
    ${PRISM_JAVA_HOME}/include/linux)
  target_compile_definitions(${name} PRIVATE
    PRISM_HIP_ARCH="${PRISM_HIP_ARCH}"
    PRISM_CHECKED=$<BOOL:${checked}>)
  # 積と和をコンパイラーに 1 つの積和の命令へまとめさせない。積和はソースが fusedMultiplyAdd で書く。
  target_compile_options(${name} PRIVATE -O3 -ffp-contract=off -Wall -Wextra -fvisibility=hidden -fvisibility-inlines-hidden)
  set_target_properties(${name} PROPERTIES HIP_ARCHITECTURES ${PRISM_HIP_ARCH} POSITION_INDEPENDENT_CODE ON)
endfunction()

add_library(prism_hip_inference_objects OBJECT ${PRISM_INFERENCE_OBJECT_SOURCES})
prism_hip_compile(prism_hip_inference_objects OFF)

function(prism_hip_library name checked)
  add_library(${name} SHARED ${PRISM_LIBRARY_SOURCES} ${ARGN})
  prism_hip_compile(${name} ${checked})
  # DJL がロードした HIP ランタイムを使うため、ROCm の RUNPATH を追加しない。
  target_link_options(${name} PRIVATE -fno-rtlib-add-rpath)
  target_link_libraries(${name} PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PRISM_OUTPUT_DIR})
endfunction()

prism_hip_library(prism_hip OFF $<TARGET_OBJECTS:prism_hip_inference_objects>)
prism_hip_library(prism_hip_check ON ${PRISM_INFERENCE_OBJECT_SOURCES})


configure_file(prism-hip.properties.in ${PRISM_OUTPUT_DIR}/prism-hip.properties @ONLY)
