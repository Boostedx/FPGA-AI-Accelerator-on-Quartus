#ifndef DEFINES_H_
#define DEFINES_H_

/*
 * Intel HLS makes use of three streaming interfaces:
 *   (1) stream_in - used as the main input to a component
 *   (2) stream_out - used as the main output of a component
 *   (3) stream - allows both reading and writing; used for inter-component connections
 * ihc::stream has a implicitly deleted constructor and therefore, cannot be used as the output of a function/component
 * Therefore, variables of type 'stream' are always passed by reference
 */

#ifndef __INTELFPGA_COMPILER__

#include "ac_fixed.h"
#include "ac_int.h"
#define hls_register

#include "stream.h"
template <typename T> using stream = nnet::stream<T>;
template <typename T> using stream_in = nnet::stream<T>;
template <typename T> using stream_out = nnet::stream<T>;

#else

#include "HLS/ac_fixed.h"
#include "HLS/ac_int.h"
#include "HLS/hls.h"

template <typename T> using stream = ihc::stream<T>;
template <typename T> using stream_in = ihc::stream_in<T>;
template <typename T> using stream_out = ihc::stream_out<T>;

#endif

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 28
#define N_INPUT_2_1 28
#define N_INPUT_3_1 1
#define N_SIZE_0_2 784
#define N_LAYER_3 16
#define N_LAYER_3 16
#define N_LAYER_6 8
#define N_LAYER_6 8
#define N_LAYER_9 10
#define N_LAYER_9 10

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ac_fixed<8,1,true>, 1*1> input_t;
typedef ac_fixed<8,1,true> model_default_t;
typedef nnet::array<ac_fixed<8,1,true>, 16*1> layer3_t;
typedef ac_fixed<8,1,true> weight3_t;
typedef ac_fixed<8,1,true> bias3_t;
typedef ac_int<1, false> layer3_index;
typedef nnet::array<ac_fixed<8,1,true>, 16*1> layer5_t;
typedef ac_fixed<18,8,true> activation_table_t;
typedef nnet::array<ac_fixed<8,1,true>, 8*1> layer6_t;
typedef ac_fixed<8,1,true> weight6_t;
typedef ac_int<1, false> layer6_index;
typedef nnet::array<ac_fixed<8,1,true>, 8*1> layer8_t;
typedef ac_fixed<18,8,true> activation_1_table_t;
typedef nnet::array<ac_fixed<8,1,true>, 10*1> layer9_t;
typedef ac_fixed<8,1,true> weight9_t;
typedef ac_int<1, false> layer9_index;
typedef nnet::array<ac_fixed<8,1,true>, 10*1> result_t;
typedef ac_fixed<18,8,true> activation_2_table_t;
typedef ac_fixed<18,8,true,AC_RND,AC_SAT> activation_2_exp_table_t;
typedef ac_fixed<18,8,true,AC_RND,AC_SAT> activation_2_inv_table_t;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
