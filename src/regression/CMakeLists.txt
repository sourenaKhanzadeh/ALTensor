
add_library(regression SHARED STATIC
LR.cpp )


install(TARGETS regression DESTINATION ${DIVISIBLE_INSTALL_LIB_DIR})