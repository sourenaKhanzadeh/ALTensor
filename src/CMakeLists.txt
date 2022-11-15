

add_library(srclib SHARED STATIC
            deriv.cpp )


install(TARGETS srclib DESTINATION ${DIVISIBLE_INSTALL_LIB_DIR})