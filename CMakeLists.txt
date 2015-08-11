cmake_minimum_required(VERSION 2.8)

function(find_cudnn)
  set(CUDNN_ROOT "" CACHE PATH "cuDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/include
    DOC "cuDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
    DOC "cuDNN library path")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found cuDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find cuDNN in path: ${CUDNN_ROOT}")
  endif()
endfunction()

function(find_blas)
  set(BLAS_ROOT "" CACHE PATH "BLAS root path")
  find_path(CBLAS_INCLUDE_DIRS cblas.h
    PATHS ${BLAS_ROOT}
          ${BLAS_ROOT}/include
    DOC "BLAS include path")
  find_library(CBLAS_LIBRARIES NAMES libcblas.so
    PATHS ${BLAS_ROOT}
          ${BLAS_ROOT}/lib
          ${BLAS_ROOT}/lib64
    DOC "BLAS library path")
  if(CBLAS_INCLUDE_DIRS AND CBLAS_LIBRARIES)
    set(CBLAS_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found BLAS (include: ${CBLAS_INCLUDE_DIRS}, library: ${CBLAS_LIBRARIES})")
    mark_as_advanced(CBLAS_INCLUDE_DIRS CBLAS_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find BLAS in path: ${BLAS_ROOT}. If you are using other BLAS than cBLAS, please link your .so into libcblas.so.")
  endif()
endfunction()

function(check_mpi_cuda_support)  
    EXECUTE_PROCESS(COMMAND ompi_info "--parsable" "-l" "9" "--all"
                    COMMAND grep "mpi_built_with_cuda_support:value"
                    COMMAND awk "BEGIN {FS = \":\"} ; {print $7}"
                    OUTPUT_VARIABLE TEMP)            
    if(TEMP EQUAL "true")
        set(MPI_CUDA_SUPPORT 1)
    else(TEMP EQUAL "true")
        message(FATAL_ERROR "MPI implementation is not cuda-aware")
    endif(TEMP EQUAL "true")
endfunction()

function(find_ps)
  set(PS_ROOT "" CACHE PATH "minervaps root path")
  find_library(PS_LIBRARIES NAMES libminervaps.a
    PATHS ${PS_ROOT}/build
    DOC "PS library path")
  if(PS_LIBRARIES)
    set(PS_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found PS (library: ${PS_LIBRARIES})")
    mark_as_advanced(PS_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find PS in path: ${PS_ROOT}. Make sure you have built ps with 'make minerva'.")
  endif()
endfunction()

project(Minerva)

function(check_rdma_support)
    #TODO(jlovitt): mpi: ompi_info --all | grep btl_openib_have_cuda_gdr ==>   MCA btl: informational "btl_openib_have_cuda_gdr" (current value: "true", data source: default, level: 4 tuner/basic, type: bool)
    #               OFED:   ompi_info --all | grep btl_openib_have_driver_gdr  ==>   MCA btl: informational "btl_openib_have_driver_gdr" (current value: "true", data source: default, level: 4 tuner/basic, type: bool)  
endfunction()

option(BUILD_CXX_APPS    "build C++ applications"                OFF)
option(BUILD_TESTS       "build all unittests"                   OFF)
option(BUILD_CPU_ONLY    "build cpu-only version"                OFF)
option(BUILD_WITH_PS     "build with parameter server support"   OFF)
option(BUILD_WITH_BLAS   "build with BLAS library for CPU"       OFF)
option(BUILD_WITH_MPI    "build with MPI distributed support"    OFF)
option(BUILD_WITH_FPGA   "build with FPGA support"               OFF)

message(STATUS "cmake generator: ${CMAKE_GENERATOR}")
message(STATUS "cmake build tool: ${CMAKE_BUILD_TOOL}")
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

############################################# compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11"       SUPPORTS_CXX11)
check_cxx_compiler_flag("-flto"            SUPPORTS_LTO)
check_cxx_compiler_flag("-mssse3"          SUPPORTS_MSSSE3)
check_cxx_compiler_flag("-ftree-vectorize" SUPPORTS_VECTORIZE)

set(COMPILER_FLAGS "-Wall -fPIC -march=native")
set(GENERAL_FLAGS_DEBUG "${COMPILER_FLAGS} -O0 -g")
set(GENERAL_FLAGS_RELEASE "${COMPILER_FLAGS} -O2 -mtune=native -mssse3")
set(CMAKE_CXX_FLAGS_DEBUG "${GENERAL_FLAGS_DEBUG} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${GENERAL_FLAGS_RELEASE} -std=c++11 -DNDEBUG")

############################################# find packages
set(CORE_DEPS "")

find_package(Threads REQUIRED)
set(CORE_DEPS ${CORE_DEPS} ${CMAKE_THREAD_LIBS_INIT})

if (BUILD_CPU_ONLY)
  message(STATUS "build cpu-only version; cuda is not enabled!")
else(BUILD_CPU_ONLY)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
  find_package(CUDA)
  if (CUDA_FOUND)
    if (CUDA_VERSION_MAJOR LESS 6)
      message(FATAL_ERROR "Need CUDA 6 for uniform addressing")
    endif (CUDA_VERSION_MAJOR LESS 6)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    add_definitions(-DHAS_CUDA)
    string(REPLACE " " "\\\",\\\"" CUDA_CONCAT_DEBUG "${GENERAL_FLAGS_DEBUG}")
    string(REPLACE " " "\\\",\\\"" CUDA_CONCAT_RELEASE "${GENERAL_FLAGS_RELEASE}")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS_DEBUG "-G -Xcompiler \\\"${CUDA_CONCAT_DEBUG}\\\"")
    set(CUDA_NVCC_FLAGS_RELEASE "-Xcompiler \\\"${CUDA_CONCAT_RELEASE}\\\"")
    set(CUDA_NVCC_FLAGS "-arch sm_35")
    find_cudnn()
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
  else ()
    message(FATAL_ERROR "CUDA enabled but not found")
  endif ()
endif(BUILD_CPU_ONLY)

############################################# third party libraries
add_subdirectory(third_party)
include_directories(SYSTEM ${THIRD_INCLUDE_PATH})
set(CORE_DEPS ${CORE_DEPS} ${THIRD_LIBS})
############################################# macros

#TODO: Fill in with fpga resource detection, includes, and linker flags.
if(BUILD_WITH_FPGA)
  add_definitions(-DHAS_FPGA)
endif()
if(BUILD_WITH_MPI)
	find_package(MPI REQUIRED)
endif(BUILD_WITH_MPI)

if(BUILD_WITH_MPI AND MPI_CXX_FOUND)
  #TODO(jlovitt): Check for cuda support in mpi
  if (CUDA_FOUND)
      check_mpi_cuda_support()
      check_rdma_support()
      if(RDMA_SUPPORT_FOUND)
          #TODO(jlovitt):notify or set mpiexec to use: --mca btl_openib_want_cuda_gdr 1
      endif(RDMA_SUPPORT_FOUND)
  endif(CUDA_FOUND)
  add_definitions(-DHAS_MPI)
  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
  include_directories(SYSTEM ${MPI_HEADER_ROOT})
  #target_link_libraries(Minerva ${MPI_CXX_LIBRARIES})
endif()

if(FIXED_POINT)
    add_definitions(-DFIXED_POINT)
    add_definitions(-DFIXED_POINT_DOUBLE_WIDE_TYPE=${FIXED_POINT_DOUBLE_WIDE_TYPE})
    add_definitions(-DFIXED_POINT_TYPE=${FIXED_POINT_TYPE})
    add_definitions(-DFIXED_POINT_FRACTION_WIDTH_N=${FIXED_POINT_FRACTION_WIDTH_N})
endif()

if(USE_PROFILER)
    add_definitions(-DUSE_PROFILER)
endif()

if(BUILD_WITH_PS)
  add_definitions(-DHAS_PS)
  find_ps()
  message(STATUS "PS enabled, forcing BUILD_TESTS to 0")
  set(BUILD_TESTS 0)
endif()

if(BUILD_WITH_BLAS)
  add_definitions(-DHAS_CBLAS)
  find_blas()
  include_directories(SYSTEM ${CBLAS_INCLUDE_DIRS})
endif()

message(STATUS "build C++ applications              -- ${BUILD_CXX_APPS}")
message(STATUS "build unit tests                    -- ${BUILD_TESTS}")
message(STATUS "build cpu-only version              -- ${BUILD_CPU_ONLY}")
message(STATUS "build with parameter server support -- ${BUILD_WITH_PS}")
message(STATUS "build with BLAS library for CPU     -- ${BUILD_WITH_BLAS}")
message(STATUS "build with MPI support              -- ${BUILD_WITH_MPI}")
message(STATUS "MPI libraries found.                -- ${MPI_CXX_FOUND}")
message(STATUS "build with FPGA support             -- ${BUILD_WITH_FPGA}")
message(STATUS "FPGA model type                     -- ${HT_MODEL_TYPE}")
message(STATUS "build with fixed point support      -- ${FIXED_POINT}")
if(FIXED_POINT)
    message(STATUS "  fixed point type.             -- ${FIXED_POINT_TYPE}")
    message(STATUS "  fixed point 2x type           -- ${FIXED_POINT_DOUBLE_WIDE_TYPE}")
    message(STATUS "  fraction width                -- ${FIXED_POINT_FRACTION_WIDTH}")
endif()

add_subdirectory(minerva)

if(BUILD_CXX_APPS)
  add_subdirectory(apps)
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
