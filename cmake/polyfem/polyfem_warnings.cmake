################################################################################
# See comments and discussions here:
# http://stackoverflow.com/questions/5088460/flags-to-enable-thorough-and-verbose-g-warnings
################################################################################

if(TARGET polyfem::warnings)
    return()
endif()

set(POLYFEM_WARNING_FLAGS
    -Wall
    -Wextra
    -pedantic

    # -Wconversion
    #-Wunsafe-loop-optimizations # broken with C++11 loops
    -Wunused

    -Wno-long-long
    -Wpointer-arith
    -Wformat=2
    -Wuninitialized
    -Wcast-qual
    # -Wmissing-noreturn
    -Wmissing-format-attribute
    -Wredundant-decls

    -Werror=implicit
    -Werror=nonnull
    -Werror=init-self
    -Werror=main
    -Werror=missing-braces
    -Werror=sequence-point
    -Werror=return-type
    -Werror=trigraphs
    -Warray-bounds
    -Werror=write-strings
    -Werror=address
    -Werror=int-to-pointer-cast
    -Werror=pointer-to-int-cast

    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-unused-parameter

    #-Weffc++
    -Wno-old-style-cast
    # -Wno-sign-conversion
    #-Wsign-conversion

    # -Wshadow

    -Wstrict-null-sentinel
    -Woverloaded-virtual
    -Wsign-promo
    -Wstack-protector
    -Wstrict-aliasing
    -Wstrict-aliasing=2
    -Wswitch-default
    # -Wswitch-enum
    -Wswitch-unreachable

    -Wcast-align
    -Wdisabled-optimization
    #-Winline # produces warning on default implicit destructor
    -Winvalid-pch
    # -Wmissing-include-dirs
    -Wpacked
    -Wno-padded
    -Wstrict-overflow
    -Wstrict-overflow=2

    -Wctor-dtor-privacy
    -Wlogical-op
    # -Wnoexcept
    -Woverloaded-virtual
    # -Wundef

    -Wnon-virtual-dtor
    -Wdelete-non-virtual-dtor
    -Werror=non-virtual-dtor
    -Werror=delete-non-virtual-dtor

    -Wno-sign-compare

    -Wsuggest-override

    ###########
    # GCC 6.1 #
    ###########

    -Wnull-dereference
    -fdelete-null-pointer-checks
    -Wduplicated-cond
    -Wmisleading-indentation

    #-Weverything

    ###########################
    # Enabled by -Weverything #
    ###########################

    #-Wdocumentation
    #-Wdocumentation-unknown-command
    #-Wfloat-equal
    #-Wcovered-switch-default

    #-Wglobal-constructors
    #-Wexit-time-destructors
    #-Wmissing-variable-declarations
    #-Wextra-semi
    #-Wweak-vtables
    #-Wno-source-uses-openmp
    #-Wdeprecated
    #-Wnewline-eof
    #-Wmissing-prototypes

    #-Wno-c++98-compat
    #-Wno-c++98-compat-pedantic

    ################################################
    # Need to check if those are still valid today #
    ################################################

    #-Wimplicit-atomic-properties
    #-Wmissing-declarations
    #-Wmissing-prototypes
    #-Wstrict-selector-match
    #-Wundeclared-selector
    #-Wunreachable-code

    # Not a warning, but enable link-time-optimization
    # TODO: Check out modern CMake version of setting this flag
    # https://cmake.org/cmake/help/latest/module/CheckIPOSupported.html
    #-flto

    # Gives meaningful stack traces
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls

    #####################
    # Disabled warnings #
    #####################

    -Wno-missing-noreturn
    -Wno-shadow
    -Wno-switch-enum
    -Wno-unused-command-line-argument
    -Wno-unused-function
    -Wno-unused-private-field
    -Wno-unused-lambda-capture
    -Wno-reorder-ctor
    -Wno-missing-field-initializers
)

# Flags above don't make sense for MSVC
if(MSVC)
    set(POLYFEM_WARNING_FLAGS)
endif()

add_library(polyfem_warnings INTERFACE)
add_library(polyfem::warnings ALIAS polyfem_warnings)

include(polyfem_filter_flags)
polyfem_filter_flags(POLYFEM_WARNING_FLAGS)
target_compile_options(polyfem_warnings INTERFACE ${POLYFEM_WARNING_FLAGS})