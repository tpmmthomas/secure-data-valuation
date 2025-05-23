project(cryptoTools)

include(FindPkgConfig)

get_directory_property(hasParent PARENT_DIRECTORY)

# add the source files
file(GLOB_RECURSE SRCS *.cpp *.c)
add_library(cryptoTools ${SRCS})

# make projects that include cryptoTools use this as an include folder
target_include_directories(cryptoTools PUBLIC
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
                    $<INSTALL_INTERFACE:include>)
target_include_directories(cryptoTools PUBLIC
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                    $<INSTALL_INTERFACE:include>)

if(NOT MSVC)
    target_compile_options(cryptoTools PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++17> -pthread)
    target_link_options(cryptoTools PUBLIC -pthread)

    if(ENABLE_SSE)
        target_compile_options(cryptoTools PUBLIC -maes -msse2)
    endif()
    if(ENABLE_AVX)
        target_compile_options(cryptoTools PUBLIC -msse3 -mssse3 -msse4.1 -mpclmul -mavx2)
    endif()
endif()



## Relic
###########################################################################

if (ENABLE_RELIC)
  target_link_libraries(cryptoTools relic)
endif (ENABLE_RELIC)

# libsodium
###########################################################################

if (ENABLE_SODIUM)
  target_link_libraries(cryptoTools sodium)
endif (ENABLE_SODIUM)

## WolfSSL
###########################################################################

if(ENABLE_WOLFSSL)
  target_include_directories(cryptoTools PUBLIC
                    $<BUILD_INTERFACE:${WOLFSSL_LIB_INCLUDE_DIRS}>
                    $<INSTALL_INTERFACE:>)
  target_link_libraries(cryptoTools ${WOLFSSL_LIB})
endif(ENABLE_WOLFSSL)

## Boost
###########################################################################


if(ENABLE_BOOST)
    target_include_directories(cryptoTools PUBLIC
                    $<BUILD_INTERFACE:${Boost_INCLUDE_DIR}>
                    $<INSTALL_INTERFACE:>)
    target_link_libraries(cryptoTools ${Boost_LIBRARIES})
endif()



