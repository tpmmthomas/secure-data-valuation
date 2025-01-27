cmake_minimum_required(VERSION 3.12)
project(SimplestOT C)


# cryptoTools code, see cryptoTools-license.txt for the license

    find_path(SODIUM_INCLUDE_DIRS sodium.h HINTS  "${OC_THIRDPARTY_HINT}/include")

    find_library(SODIUM_LIBRARIES NAMES sodium libsodium HINTS "${OC_THIRDPARTY_HINT}/lib")

    if (NOT SODIUM_INCLUDE_DIRS OR NOT SODIUM_LIBRARIES)
        message(FATAL_ERROR "Failed to find libsodium.\n  OC_THIRDPARTY_HINT=${OC_THIRDPARTY_HINT}\n  SODIUM_INCLUDE_DIRS=${SODIUM_INCLUDE_DIRS}\n  SODIUM_LIBRARIES=${SODIUM_LIBRARIES}")
    endif ()

    set(SODIUM_MONTGOMERY ON CACHE BOOL "SODIUM_MONTGOMERY...")

    message(STATUS "SODIUM_INCLUDE_DIRS:  ${SODIUM_INCLUDE_DIRS}")
    message(STATUS "SODIUM_LIBRARIES:  ${SODIUM_LIBRARIES}")
    message(STATUS "SODIUM_MONTGOMERY:  ${SODIUM_MONTGOMERY}\n")

    add_library(sodium STATIC IMPORTED)

    set_property(TARGET sodium PROPERTY IMPORTED_LOCATION ${SODIUM_LIBRARIES})
    target_include_directories(sodium INTERFACE
                    $<BUILD_INTERFACE:${SODIUM_INCLUDE_DIRS}>
                    $<INSTALL_INTERFACE:>)


link_libraries(sodium)

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -DREF10 -std=gnu99")

include_directories(.)

set(SRCS
    crypto_hash.h
    cpucycles.c
    fe.h
    fe.c
    fe_0.c
    fe_1.c
    fe_add.c
    fe_cmov.c
    fe_copy.c
    fe_frombytes.c
    fe_invert.c
    fe_isnegative.c
    fe_isnonzero.c
    fe_mul.c
    fe_neg.c
    fe_pow22523.c
    fe_sq.c
    fe_sq2.c
    fe_sub.c
    fe_tobytes.c
    ge.h
    ge.c
    ge_add.c
    ge_add.h
    ge_double_scalarmult.c
    ge_frombytes.c
    ge_madd.c
    ge_madd.h
    ge_msub.c
    ge_msub.h
    ge_p1p1_to_p2.c
    ge_p1p1_to_p3.c
    ge_p2_0.c
    ge_p2_dbl.c
    ge_p2_dbl.h
    ge_p2_dbl.q
    ge_p3_0.c
    ge_p3_dbl.c
    ge_p3_to_cached.c
    ge_p3_to_p2.c
    ge_p3_tobytes.c
    ge_precomp_0.c
    ge_scalarmult_base.c
    ge_sub.c
    ge_sub.h
    ge_tobytes.c
    Keccak-simple-settings.h
    Keccak-simple.c
    network.c
    network.h
    ot_config.h
    ot_receiver.c
    ot_receiver.h
    ot_sender.c
    ot_sender.h
    ../common/randombytes.c
    ../common/randombytes.h
    sc.h
    sc_muladd.c
    sc_random.c
    sc_reduce.c
    verify.c
    )
add_library(SimplestOT STATIC ${SRCS})

add_executable(ot_sender ../common/ot_sender_test.c)
target_link_libraries(ot_sender SimplestOT)

add_executable(ot_receiver ../common/ot_receiver_test.c)
target_link_libraries(ot_receiver SimplestOT)
