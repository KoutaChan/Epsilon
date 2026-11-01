# libprism_hip.so: PRISM の HIP の実行系。網を知らない JNI と自前のカーネルだけを持つ。
#
# libtorch のヘッダは使わない（サーバに libtorch 2.11.0+rocm7.2 のヘッダがない）。学習の演算が DJL の配列のアドレスを引くときだけ、
# libtorch が公開する at::TensorBase::const_data_ptr<T>() を dlsym で呼ぶ（prism_hip_torch.cc）。繋がるのは HIP の実行系（libamdhip64.so.7）だけで、
# DJL が先に読んだものを soname で使い回す。ストリームは PtStream.getId()、ホストのメモリは直の ByteBuffer で受け、デバイスの
# メモリは計画（newPlan）が一度だけ取る。RPATH は付けない。
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

set(PRISM_SOURCES
  prism_hip_jni.cc
  prism_hip_plan.cc
  prism_hip_runtime.cc
  prism_hip_torch.cc
  k0_relation.hip
  k1_embed.hip
  k2_attention.hip
  k3_mlp.hip
  k6_candidate.hip
  k7_value_graph.hip
  t0_relation.hip
  t1_attention.hip
  t2_norm.hip)
# 全ての翻訳単位を HIP として 7.2.2 の clang に通す（HIP の実行系のヘッダと定義をそろえる）。
set_source_files_properties(prism_hip_jni.cc prism_hip_plan.cc prism_hip_runtime.cc prism_hip_torch.cc PROPERTIES LANGUAGE HIP)
# 注意と学習の RMSNorm のカーネルは、レジスタ・scratch・LDS・占有を組み立ての記録に出す。
set_source_files_properties(k0_relation.hip k2_attention.hip t1_attention.hip t2_norm.hip
  PROPERTIES COMPILE_OPTIONS "-Rpass-analysis=kernel-resource-usage")

function(prism_hip_library name checked)
  add_library(${name} SHARED ${PRISM_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PRISM_JAVA_HOME}/include
    ${PRISM_JAVA_HOME}/include/linux)
  target_compile_definitions(${name} PRIVATE
    PRISM_HIP_ARCH="${PRISM_HIP_ARCH}"
    PRISM_HIP_CHECKED=$<BOOL:${checked}>
    $<$<BOOL:${checked}>:PRISM_HIP_CHECK>)
  target_compile_options(${name} PRIVATE -O3 -Wall -Wextra -fvisibility=hidden -fvisibility-inlines-hidden)
  # ROCm の clang の設定（rocm.cfg）は -frtlib-add-rpath で HIP の実行系の RUNPATH を足すので、ここで外す。
  target_link_options(${name} PRIVATE -fno-rtlib-add-rpath)
  target_link_libraries(${name} PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(${name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PRISM_OUTPUT_DIR}
    HIP_ARCHITECTURES ${PRISM_HIP_ARCH})
endfunction()

prism_hip_library(prism_hip OFF)
prism_hip_library(prism_hip_check ON)

configure_file(prism-hip.properties.in ${PRISM_OUTPUT_DIR}/prism-hip.properties @ONLY)
