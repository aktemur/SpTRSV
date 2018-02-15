cmake_minimum_required(VERSION 3.1)
project(SpTRSV)

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
                  
message(STATUS "Build type is " ${CMAKE_BUILD_TYPE})
message(STATUS "System is " ${CMAKE_SYSTEM_NAME})
message(STATUS "Compiler is " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "Source dir is " ${CMAKE_SOURCE_DIR})
message(STATUS "Source dir is " ${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 14)

message(STATUS "CXX Flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "Linker Flags: " ${CMAKE_EXE_LINKER_FLAGS})
set(dir ${CMAKE_CURRENT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}")

set(SOURCE_FILES
    main.cpp
    profiler.cpp
)

set(HEADER_FILES
    concurrentqueue.h
    europar16.hpp
    experimental.hpp
    hpcs17.hpp
    mklsolver.hpp
    profiler.h
    sequential.hpp
    csrlenForwardSolve.cpp
    solver.hpp
)

add_executable(sptrsv ${SOURCE_FILES})
target_sources(sptrsv PUBLIC ${HEADER_FILES})

##
## MMMatrixIO library
##
find_library(MMMatrixIO
  mmmatrixio
  PATHS ${CMAKE_SOURCE_DIR}/../lib/mmmatrixio
)
target_link_libraries(sptrsv ${MMMatrixIO})
target_include_directories(sptrsv PUBLIC ${CMAKE_SOURCE_DIR}/../MMMatrixIO/src)

##
## docopt library
##
include_directories(${CMAKE_SOURCE_DIR}/../docopt.cpp)
find_library(docopt
  docopt
  PATHS ${CMAKE_SOURCE_DIR}/../lib/docopt.cpp
)
target_link_libraries(sptrsv ${docopt})


##
## MKL
##
set(MKL_ROOT_DIR $ENV{MKLROOT})
if (MKL_ROOT_DIR)
  message(STATUS "Found MKL at " $ENV{MKLROOT})
  add_definitions(-DMKL_EXISTS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64" )
  include_directories(${MKL_ROOT_DIR}/include)
  find_library(MKL_CORE
    mkl_core
    PATHS ${MKL_ROOT_DIR}/lib ${MKL_ROOT_DIR}/lib/intel64
  )
  find_library(MKL_INTEL_THREAD
    mkl_intel_thread
    PATHS ${MKL_ROOT_DIR}/lib ${MKL_ROOT_DIR}/lib/intel64
  )
  find_library(MKL_INTEL_LP64
    mkl_intel_lp64
    PATHS ${MKL_ROOT_DIR}/lib ${MKL_ROOT_DIR}/lib/intel64
  )
  find_library(MKL_INTEL_LP64
    iomp5
    PATHS ${MKL_ROOT_DIR}/../lib ${MKL_ROOT_DIR}/../lib/intel64
  )
  target_link_libraries(sptrsv
    ${MKL_CORE}
    ${MKL_INTEL_THREAD}
    ${MKL_INTEL_LP64}
    ${IOMP5}
    pthread
    m
    dl
  )
endif()


##
## OpenMP
##
find_package(OpenMP)
if (OPENMP_FOUND)
  add_definitions(-DOPENMP_EXISTS)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "Compiler flags (OMP): " ${CMAKE_CXX_FLAGS})
endif()

##
## Intel TBB
##
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
find_package(TBB)
target_link_libraries(sptrsv tbb)

##
## Boost
##
#find_package(Boost)
#include_directories(${Boost_INCLUDE_DIR})
