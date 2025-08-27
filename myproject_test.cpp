#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "firmware/myproject.h"
#include "firmware/parameters.h"

#include "firmware/nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

// This function is written to avoid stringstream, which is
// not supported in cosim 20.1, and because strtok
// requires a const_cast or allocation to use with std::strings.
// This function returns the next float (by argument) at position pos,
// updating pos. True is returned if conversion done, false if the string
// has ended, and std::invalid_argument exception if the sting was bad.
bool nextToken(const std::string &str, std::size_t &pos, float &val) {
    while (pos < str.size() && std::isspace(static_cast<unsigned char>(str[pos]))) {
        pos++;
    }
    if (pos >= str.size()) {
        return false;
    }
    std::size_t offset = 0;
    val = std::stof(str.substr(pos), &offset);
    pos += offset;
    return true;
}

int main(int argc, char **argv) {
    // Load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    std::string iline;

    // Load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");
    std::string pline;

    // Output log
    std::string RESULTS_LOG = "tb_data/results.log";
    std::ofstream fout(RESULTS_LOG);

    if (fin.is_open() && fpr.is_open()) {
        std::vector<std::vector<float>> predictions;

        unsigned int iteration = 0;
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (iteration % CHECKPOINT == 0) {
                std::cout << "Processing input " << iteration << std::endl;
            }

            // hls-fpga-machine learning instantiate inputs and outputs
            stream_in<input_t> flatten_input_input;
            stream_out<result_t> layer11_out_output;

            std::vector<float> in;
            std::vector<float> pr;
            float current;

            std::size_t pos = 0;
            while (nextToken(iline, pos, current)) {
                in.push_back(current);
            }

            pos = 0;
            while (nextToken(pline, pos, current)) {
                pr.push_back(current);
            }

            // hls-fpga-machine-learning insert data
            float vals_0[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1]; 
            for (int j = 0 ; j < N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1 ; j++) {
                        vals_0[j] = in[j]; 
            }
            nnet::convert_data<float, input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1>(vals_0, flatten_input_input);

            predictions.push_back(std::move(pr));

            // hls-fpga-machine-learning insert top-level-function
            ihc_hls_enqueue_noret(&myproject, flatten_input_input, layer11_out_output);

            // hls-fpga-machine-learning insert run
            ihc_hls_component_run_all(myproject);

            // hls-fpga-machine-learning convert output
            float res[N_LAYER_9];
            nnet::convert_data_back<result_t, float, N_LAYER_9>(layer11_out_output, res);


            for(int i = 0; i < N_LAYER_9; i++) {
              fout << res[i] << " ";
            }
            fout << std::endl;

            if (iteration % CHECKPOINT == 0) {
                std::cout << "Python Predictions" << std::endl;
                // hls-fpga-machine-learning print predictions
                for(int i = 0; i < N_LAYER_9; i++) {
                  std::cout << predictions[iteration][i] << " ";
                }
                std::cout << std::endl;

                std::cout << "HLS predictions" << std::endl;
                // hls-fpga-machine-learning print output
                for(int i = 0; i < N_LAYER_9; i++) {
                  std::cout << res[i] << " "; 
                } 
                std::cout << std::endl; 
            }

            iteration++;
        }

        fin.close();
        fpr.close();

    } else {
        const unsigned int num_iterations = 10;
        std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                  << " invocations." << std::endl;

        for (int iteration = 0; iteration < num_iterations; iteration++) {
            // hls-fpga-machine learning instantiate inputs and outputs
            stream_in<input_t> flatten_input_input;
            stream_out<result_t> layer11_out_output;

            // hls-fpga-machine-learning insert zero
            float vals_0[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1]; 
            for (int j = 0 ; j < N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1 ; j++) {
                        vals_0[j] = 0.0; 
            }
            nnet::convert_data<float, input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1>(vals_0, flatten_input_input);

            // hls-fpga-machine-learning insert top-level-function
            ihc_hls_enqueue_noret(&myproject, flatten_input_input, layer11_out_output);

            // hls-fpga-machine-learning insert run
            ihc_hls_component_run_all(myproject);

            // hls-fpga-machine-learning convert output
            float res[N_LAYER_9];
            nnet::convert_data_back<result_t, float, N_LAYER_9>(layer11_out_output, res);


            for(int i = 0; i < N_LAYER_9; i++) {
              fout << res[i] << " ";
            }
            fout << std::endl;

            if (iteration % CHECKPOINT == 0) {
                std::cout << "HLS predictions" << std::endl;
                // hls-fpga-machine-learning print output
                for(int i = 0; i < N_LAYER_9; i++) {
                  std::cout << res[i] << " "; 
                } 
                std::cout << std::endl; 
            }
        }
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
