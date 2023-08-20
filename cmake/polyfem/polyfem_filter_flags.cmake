#
# Copyright 2021 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#
function(polyfem_filter_flags flags)
  include(CheckCXXCompilerFlag)
  set(output_flags)
  foreach(FLAG IN ITEMS ${${flags}})
    string(REPLACE "=" "-" FLAG_VAR "${FLAG}")
    if(NOT DEFINED IS_SUPPORTED_${FLAG_VAR})
      check_cxx_compiler_flag("${FLAG}" IS_SUPPORTED_${FLAG_VAR})
    endif()
    if(IS_SUPPORTED_${FLAG_VAR})
      list(APPEND output_flags $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif()
  endforeach()
  set(${flags} ${output_flags} PARENT_SCOPE)
endfunction()