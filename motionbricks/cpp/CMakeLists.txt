cmake_minimum_required(VERSION 3.18)
project(motionbricks_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime prefix")
if(NOT ORT_ROOT)
  set(ORT_ROOT "/home/michael/Work-syncfree/bfm_controller/third_party/onnxruntime-linux-aarch64-musa-sdk-5.1.0")
endif()
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ORT_ROOT}/include" REQUIRED)
find_library(ORT_LIBRARY NAMES onnxruntime PATHS "${ORT_ROOT}/lib" REQUIRED)
find_library(MUSART_LIBRARY NAMES musart
  PATHS "$ENV{MUSA_HOME}/lib" /usr/local/musa-5.1.0/lib /usr/local/musa/lib
  REQUIRED)

set(BFM_CONTROLLER_ROOT "/home/michael/Work-syncfree/bfm_controller"
    CACHE PATH "bfm_controller repository containing the final MotionBricks planner")
set(BFM_INCLUDE_DIR "${BFM_CONTROLLER_ROOT}/bfm_controller/include")

set(MOTIONBRICKS_PLANNER_SOURCES
  "${BFM_CONTROLLER_ROOT}/bfm_controller/src/ctrl_env/motionbricks/asset_pack.cc"
  "${BFM_CONTROLLER_ROOT}/bfm_controller/src/ctrl_env/motionbricks/motionbricks_features.cc"
  "${BFM_CONTROLLER_ROOT}/bfm_controller/src/ctrl_env/motionbricks/motionbricks_ort.cc"
  "${BFM_CONTROLLER_ROOT}/bfm_controller/src/ctrl_env/motionbricks/planner.cc"
  "${BFM_CONTROLLER_ROOT}/bfm_controller/src/engine/host_utils.cc"
  "${BFM_CONTROLLER_ROOT}/bfm_controller/src/engine/musa_utils.cc"
  "${BFM_CONTROLLER_ROOT}/bfm_controller/src/engine/ort/ort_engine.cc"
)

foreach(source IN LISTS MOTIONBRICKS_PLANNER_SOURCES)
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "Missing bfm_controller MotionBricks source: ${source}")
  endif()
endforeach()

add_executable(motionbricks_ort_cli motionbricks_ort_cli.cc)
target_include_directories(motionbricks_ort_cli PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(motionbricks_ort_cli PRIVATE "${ORT_LIBRARY}")
set_target_properties(motionbricks_ort_cli PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib;$ENV{MUSA_HOME}/lib;/usr/local/musa-5.1.0/lib")

add_library(motionbricks_final_planner STATIC ${MOTIONBRICKS_PLANNER_SOURCES})
target_compile_definitions(motionbricks_final_planner PUBLIC BFM_ENABLE_AOT=0)
target_include_directories(motionbricks_final_planner PUBLIC
  "${BFM_INCLUDE_DIR}"
  "${ORT_INCLUDE_DIR}"
  /usr/local/musa/include)
target_link_libraries(motionbricks_final_planner PUBLIC
  "${ORT_LIBRARY}"
  "${MUSART_LIBRARY}")

add_executable(motionbricks_planner_cli motionbricks_planner_cli.cc)
target_link_libraries(motionbricks_planner_cli PRIVATE motionbricks_final_planner)
set_target_properties(motionbricks_planner_cli PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib;$ENV{MUSA_HOME}/lib;/usr/local/musa-5.1.0/lib")
