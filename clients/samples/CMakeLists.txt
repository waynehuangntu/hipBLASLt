# ########################################################################
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Samples
add_executable( sample_hipblaslt_gemm_tuning_splitk_ext 03_splitk_gemm_ext/sample_hipblaslt_gemm_tuning_splitk_ext.cpp)
add_executable( sample_hipblaslt_gemm_get_all_algos 05_get_all_algos_gemm/sample_hipblaslt_gemm_get_all_algos.cpp)
add_executable( sample_hipblaslt_gemm_get_all_algos_ext 05_get_all_algos_gemm_ext/sample_hipblaslt_gemm_get_all_algos_ext.cpp)
add_executable( sample_hipblaslt_gemm_get_algo_by_index_ext 06_get_algo_by_index_gemm_ext/sample_hipblaslt_gemm_get_algo_by_index_ext.cpp)
add_executable( sample_hipblaslt_gemm_gelu_aux_bias 09_gemm_gelu_aux_bias/sample_hipblaslt_gemm_gelu_aux_bias.cpp)
add_executable( sample_hipblaslt_gemm_gelu_aux_bias_ext 09_gemm_gelu_aux_bias_ext/sample_hipblaslt_gemm_gelu_aux_bias_ext.cpp)
add_executable( sample_hipblaslt_gemm_dgelu_bgrad 13_gemm_dgelu_bgrad/sample_hipblaslt_dgelu_bgrad.cpp)
add_executable( sample_hipblaslt_gemm_dgelu_bgrad_ext 13_gemm_dgelu_bgrad_ext/sample_hipblaslt_gemm_dgelu_bgrad_ext.cpp)
add_executable( sample_hipblaslt_gemm_with_scale_a_b 16_gemm_scale_a_b/sample_hipblaslt_gemm_with_scale_a_b.cpp)
add_executable( sample_hipblaslt_gemm_with_scale_a_b_ext 16_gemm_scale_a_b_ext/sample_hipblaslt_gemm_with_scale_a_b._ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_ext 17_gemm_grouped_ext/sample_hipblaslt_groupedgemm_ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_fixed_mk_ext 18_fixed_mk_gemm_grouped_ext/sample_hipblaslt_groupedgemm_fixed_mk_ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_get_all_algos_ext 19_get_all_algos_gemm_grouped_ext/sample_hipblaslt_groupedgemm_get_all_algos_ext.cpp)
add_executable( sample_hipblaslt_gemm_attr_tciA_tciB 22_gemm_cvtA_cvtB/sample_hipblaslt_gemm_attr_tciA_tciB.cpp)
add_executable( sample_hipblaslt_ext_op_layernorm 23_layernorm_ext/sample_hipblaslt_ext_op_layernorm.cpp)
add_executable( sample_hipblaslt_gemm_ext_deprecated deprecated/sample_hipblaslt_gemm_ext_deprecated.cpp)

set(samples sample_hipblaslt_gemm_tuning_splitk_ext
            sample_hipblaslt_gemm_get_all_algos
            sample_hipblaslt_gemm_get_all_algos_ext
            sample_hipblaslt_gemm_get_algo_by_index_ext
            sample_hipblaslt_gemm_gelu_aux_bias
            sample_hipblaslt_gemm_gelu_aux_bias_ext
            sample_hipblaslt_gemm_dgelu_bgrad
            sample_hipblaslt_gemm_dgelu_bgrad_ext
            sample_hipblaslt_gemm_with_scale_a_b
            sample_hipblaslt_gemm_with_scale_a_b_ext
            sample_hipblaslt_groupedgemm_ext
            sample_hipblaslt_groupedgemm_fixed_mk_ext
            sample_hipblaslt_groupedgemm_get_all_algos_ext
            sample_hipblaslt_gemm_attr_tciA_tciB
            sample_hipblaslt_ext_op_layernorm
            sample_hipblaslt_gemm_ext_deprecated
            )

set( sample_list_all ${samples})

set( sample_list_hip_device ${sample_list_all} )

foreach( exe ${sample_list_all} )
  target_link_libraries( ${exe} PRIVATE roc::hipblaslt )

  set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
  )

  target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      )

  if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c -Wno-unused-result )
    target_compile_definitions( ${exe} PRIVATE ROCBLASLT_INTERNAL_API )
  endif( )
endforeach( )

foreach( exe ${sample_list_hip_device} )
  target_link_libraries( ${exe} PRIVATE hip::device )
endforeach( )

foreach( exe ${sample_list_all} )
  rocm_install(TARGETS ${exe} COMPONENT samples)
endforeach( )

# Add tests for samples
foreach( exe ${samples} )
  add_test(NAME ${exe} COMMAND ${exe})
endforeach()
