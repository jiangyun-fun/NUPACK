cmake_minimum_required(VERSION 3.10)
################################################################################

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

################################################################################

enable_language(CXX)

################################################################################
project(NUPACK-solver VERSION 1.0.0.0 LANGUAGES CXX)
################################################################################

add_library(nupack-solver STATIC ${CMAKE_CURRENT_SOURCE_DIR}/source/Concentration.cc)
target_include_directories(nupack-solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(nupack-solver PUBLIC cxx_std_14)

find_package(fmt CONFIG REQUIRED)
if(NUPACK_EXTERNAL_ARMADILLO)
  message("-- Using external armadillo for NUPACK solver from ${ARMADILLO_INCLUDE}")
  target_include_directories(nupack-solver PUBLIC ${ARMADILLO_INCLUDE})
  target_compile_definitions(nupack-solver PUBLIC ARMA_DONT_USE_WRAPPER=1)
  if(NUPACK_ARMADILLO_NOBLAS)
    target_compile_definitions(nupack-solver PRIVATE ARMA_DONT_USE_BLAS=1)
  endif()
else()
  message("-- Using internal armadillo for NUPACK solver")
  find_package(Armadillo CONFIG)
endif()
message("-- Linking armadillo for NUPACK solver from ${ARMADILLO_LIBRARIES}")
target_link_libraries(nupack-solver PUBLIC ${ARMADILLO_LIBRARIES} fmt::fmt)

target_compile_features(nupack-solver PUBLIC cxx_std_14)

################################################################################
