cmake_minimum_required(VERSION 3.20)
project(prism_inference_tests LANGUAGES CXX)
enable_testing()
set(inference "${CMAKE_CURRENT_SOURCE_DIR}/../../main/native/inference")
add_executable(inference_plan_test inference_plan_test.cc
  "${inference}/declared_plan.cc" "${inference}/execution_plan.cc" "${inference}/workspace_layout.cc" "${inference}/packed_weights.cc")
target_compile_features(inference_plan_test PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(inference_plan_test PRIVATE /UNDEBUG /utf-8)
else()
  target_compile_options(inference_plan_test PRIVATE -UNDEBUG)
endif()
target_include_directories(inference_plan_test PRIVATE "${inference}/.." "${inference}")
add_test(NAME inference_plan COMMAND inference_plan_test)

set(host_submission "${inference}/batch_executor.cu" "${inference}/kernel_selection.cu")
set_source_files_properties(${host_submission} PROPERTIES LANGUAGE CXX)
add_executable(collection_submission_test
  collection_submission_test.cc recorded_commands.cc
  recording_backend/runtime.cc recording_backend/kernel_calls.cc
  ${host_submission} "${inference}/../cache/candidate_offset_cache.cc" "${inference}/declared_plan.cc"
  "${inference}/execution_plan.cc"
  "${inference}/workspace_layout.cc" "${inference}/packed_weights.cc")
target_include_directories(collection_submission_test PRIVATE recording_backend .
  "${inference}/.." "${inference}")
target_compile_features(collection_submission_test PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(collection_submission_test PRIVATE /UNDEBUG /utf-8)
else()
  target_compile_options(collection_submission_test PRIVATE -UNDEBUG)
endif()
add_test(NAME collection_submission COMMAND collection_submission_test)

add_executable(collection_capacity_test collection_capacity_test.cc recorded_commands.cc
  recording_backend/runtime.cc recording_backend/kernel_calls.cc
  ${host_submission} "${inference}/../cache/candidate_offset_cache.cc" "${inference}/declared_plan.cc"
  "${inference}/execution_plan.cc" "${inference}/workspace_layout.cc" "${inference}/packed_weights.cc")
target_include_directories(collection_capacity_test PRIVATE recording_backend .
  "${inference}/.." "${inference}")
target_compile_features(collection_capacity_test PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(collection_capacity_test PRIVATE /UNDEBUG /utf-8)
else()
  target_compile_options(collection_capacity_test PRIVATE -UNDEBUG)
endif()
add_test(NAME collection_capacity COMMAND collection_capacity_test)

foreach(test event_cache_test training_gradients_test)
  add_executable(${test} ${test}.cc)
  target_compile_features(${test} PRIVATE cxx_std_17)
  target_include_directories(${test} PRIVATE ../../main/native)
  if(MSVC)
    target_compile_options(${test} PRIVATE /UNDEBUG /utf-8)
  else()
    target_compile_options(${test} PRIVATE -UNDEBUG)
  endif()
  add_test(NAME ${test} COMMAND ${test})
endforeach()

add_executable(declaration_round_trip declaration_round_trip.cc)
target_compile_features(declaration_round_trip PRIVATE cxx_std_17)
target_include_directories(declaration_round_trip PRIVATE ../../main/native)
set_target_properties(declaration_round_trip PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_BINARY_DIR}/bin>")
