#include "myproject.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w9.h"
#include "weights/b9.h"

/*
 * Intel HLS requires that all 'stream' types are:
 *     (1) Passed by reference to the top-level entity or
 *     (2) Declared as global variables, outside of the main function
 * Therefore, layer inputs/output (connections betweenn individual layers) are declared here
 */
// hls-fpga-machine-learning insert inter-task streams
stream<input_t> flatten_input;
auto& layer2_out = flatten_input;
stream<layer3_t> layer3_out;
stream<layer5_t> layer5_out;
stream<layer6_t> layer6_out;
stream<layer8_t> layer8_out;
stream<layer9_t> layer9_out;
stream<result_t> layer11_out;

#ifndef __INTELFPGA_COMPILER__
/*
* The top-level function used during GCC compilation / hls4ml.predic(...) goes here
* An important distinction is made between io_stream and io_parallel:
*     (1) io_parallel:
               - Top-level function takes a struct containing an array as function argument
               - Returns a struct containing an array - the prediction
      (2) io_stream:
               - Top-level function is 'void' - no return value
               - Instead, both the input and output are passed by reference
               - This is due the HLS Streaming Interfaces; stream cannot be copied (implicitly deleted copy constructor)
* This distinction is handled in quartus_writer.py
*/
// hls-fpga-machine-learning instantiate GCC top-level
void myproject(
   stream_in<input_t> &flatten_input_stream,
   stream_out<result_t> &layer11_out_stream
) {
#else
// Maximum initiation interval, concurrency and frequency for HLS syntheis are defined here
// hls-fpga-machine-learning insert cpragmas
hls_scheduler_target_fmax_mhz(20)

/*
 * The top-level function used during HLS Synthesis goes here
 * In a similar manner to GCC, there is a distinction between io_stream & io_parallel
 */
// hls-fpga-machine-learning instantiate HLS top-level
component void myproject(
   stream_in<input_t> &flatten_input_stream,
   stream_out<result_t> &layer11_out_stream
) {
#endif
// If using io_parallel, the output needs to be initialised and returned at the end of this function
// If using io_stream, no output is initialised, as it is passed by reference to the top-level function
// hls-fpga-machine-learning initialize input/output
   for (size_t i = 0; i < N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1 / input_t::size; i++) {
     input_t tmp = flatten_input_stream.read();
     flatten_input.write(tmp);
   }

// ****************************************
// NETWORK INSTANTIATION
// ****************************************

// hls-fpga-machine-learning insert layers

    nnet::dense_resource<input_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3);

    nnet::relu<layer3_t, layer5_t, relu_config5>(layer3_out, layer5_out);

    nnet::dense_resource<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

    nnet::relu<layer6_t, layer8_t, relu_config8>(layer6_out, layer8_out);

    nnet::dense_resource<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9);

    nnet::softmax<layer9_t, result_t, softmax_config11>(layer9_out, layer11_out);


// hls-fpga-machine-learning return
   for (size_t i = 0; i < N_LAYER_9 / result_t::size; i++) {
     result_t tmp = layer11_out.read();
     layer11_out_stream.write(tmp);
   }
}
