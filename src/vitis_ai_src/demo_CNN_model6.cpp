/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include<vector>
using namespace std;
using namespace cv;

// The parameters of yolov3_voc, each value could be set as actual needs.
// Such format could be refer to the prototxts in /etc/dpu_model_param.d.conf/.
const string cnn_model6_int_config = {
    "   name: \"CNN_model6_int\" \n"
    "   model_type: CLASSIFICATION  \n"
    "   cnn_param { \n"
    "     num_classes: 2 \n"
    "     input_layers: [ \n"
    "       { \n"
    "         name: \"input_layer\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 8 \n"
    "           kernel_size: 5 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "         pool_type: \"max\" \n"
    "         pool_size: 2 \n"
    "         pool_stride: 2 \n"
    "       } \n"
    "     ] \n"
    "     conv_layers: [ \n"
    "       { \n"
    "         name: \"conv_layer1\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 12 \n"
    "           kernel_size: 1 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "       }, \n"
    "       { \n"
    "         name: \"conv_layer2\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 12 \n"
    "           kernel_size: 3 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "         pool_type: \"max\" \n"
    "         pool_size: 2 \n"
    "         pool_stride: 2 \n"
    "       }, \n"
    "       { \n"
    "         name: \"conv_layer3\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 30 \n"
    "           kernel_size: 1 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "       }, \n"
    "       { \n"
    "         name: \"conv_layer4\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 30 \n"
    "           kernel_size: 3 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "         pool_type: \"max\" \n"
    "         pool_size: 2 \n"
    "         pool_stride: 2 \n"
    "       }, \n"
    "       { \n"
    "         name: \"conv_layer5\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 16 \n"
    "           kernel_size: 1 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "       }, \n"
    "       { \n"
    "         name: \"conv_layer6\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 16 \n"
    "           kernel_size: 3 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "         pool_type: \"max\" \n"
    "         pool_size: 2 \n"
    "         pool_stride: 2 \n"
    "       }, \n"
    "       { \n"
    "         name: \"conv_layer7\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 8 \n"
    "           kernel_size: 1 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "       }, \n"
    "       { \n"
    "         name: \"conv_layer8\" \n"
    "         type: \"conv\" \n"
    "         conv_param { \n"
    "           num_output: 8 \n"
    "           kernel_size: 3 \n"
    "           stride: 1 \n"
    "           padding: 0 \n"
    "         } \n"
    "         activation: \"ReLU\" \n"
    "         batch_norm: true \n"
    "         pool_type: \"max\" \n"
    "         pool_size: 2 \n"
    "         pool_stride: 2 \n"
    "       } \n"
    "     ] \n"
    "     class_layers: [ \n"
    "       { \n"
    "         name: \"fully_connected\" \n"
    "         type: \"fc\" \n"
    "         dropout: 0.2 \n"
    "         num_output: 2 \n"
    "       } \n"
    "     ] \n"
    "   } \n"
    "}\n"
};
/*
const string yolov3_config = {
    "   name: \"CNN_model6\" \n"
    "   model_type : CNN_model6 \n"
    "   CNN_model6_param { \n"
    "     num_classes: 2 \n"
    "     anchorCnt: 3 \n"
    "     conf_threshold: 0.3 \n"
    "     nms_threshold: 0.45 \n"
    "     layer_name: \"81\" \n"
    "     layer_name: \"93\" \n"
    "     layer_name: \"105\" \n"
    "     biases: 10 \n"
    "     biases: 13 \n"
    "     biases: 16 \n"
    "     biases: 30 \n"
    "     biases: 33 \n"
    "     biases: 23 \n"
    "     biases: 30 \n"
    "     biases: 61 \n"
    "     biases: 62 \n"
    "     biases: 45 \n"
    "     biases: 59 \n"
    "     biases: 119 \n"
    "     biases: 116 \n"
    "     biases: 90 \n"
    "     biases: 156 \n"
    "     biases: 198 \n"
    "     biases: 373 \n"
    "     biases: 326 \n"
    "     test_mAP: false \n"
    "   } \n"};
*/
// Assuming output_data is a pointer to the raw output data and output_size is the number of elements
std::vector<float> softmax2(const float* logits, size_t size) {
    std::vector<float> exp_values(size);
    float max_logit = *std::max_element(logits, logits + size);

    // Compute exponentials of logits
    for (size_t i = 0; i < size; ++i) {
        exp_values[i] = std::exp(logits[i] - max_logit);  // subtract max_logit for numerical stability
    }
	
    // Compute the sum of exponentials
    float sum_exp = std::accumulate(exp_values.begin(), exp_values.end(), 0.0f);

    // Normalize to get probabilities
    for (size_t i = 0; i < size; ++i) {
        exp_values[i] /= sum_exp;
    }

    return exp_values;
}
// Function to compute the softmax of a vector
std::vector<float> softmax(const std::vector<float>& input) {
    // Step 1: Compute the maximum value in the input vector
    float max_val = *max_element(input.begin(), input.end());

    // Step 2: Compute the sum of exp(input[i] - max_val)
    float sum = 0.0;
    std::vector<float> exp_values(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        exp_values[i] = std::exp(input[i] - max_val);
        sum += exp_values[i];
    }

    // Step 3: Divide each exp_value by the sum to get the softmax output
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp_values[i] / sum;
    }

    return output;
}
/*
static std::vector<float> convert_fixpoint_to_float(const xilinx::ai::OutputTensor &tensor) {
    auto scale = xilinx::ai::tensor_scale(tensor);
    auto data = (signed char *)tensor.data;
    auto size = tensor.width * tensor.height * tensor.channel;

    auto ret = std::vector<float>(size);
    std::transform(data, data + size, ret.begin(), [scale](signed char v) {
        return ((float)v) * scale;
    });

    return ret;
}

static std::vector<std::pair<int, float>> post_process(const xilinx::ai::OutputTensor &tensor) {
    // run softmax
    auto softmax_input = convert_fixpoint_to_float(tensor);
    auto softmax_output = softmax(softmax_input);

    constexpr int TOPK = 1;
    return topk(softmax_output, TOPK);
}
*/


int main(int argc, char* argv[]) {

  	if (argc != 4) {
		std::cerr << "Please input a model name as the first param!" << std::endl;
        std::cerr << "Please input your image path as the second param!" << std::endl;
        std::cerr << "The third param is a txt to store results!" << std::endl;
        return -1; // Exit with error code
    }
    // Initialize Google Logging
    google::InitGoogleLogging(argv[0]);
    
  	//string model = argv[1] + string("_acc");
  	cv::String path = argv[2];
  	std::ofstream out_fs(argv[3], std::ofstream::out);
  	int length = path.size();
  
  	// A kernel name, it should be samed as the dnnc result. e.g.
  	// /usr/share/vitis_ai_library/models/yolov3_voc/yolov3_voc.elf
  	auto kernel_name = argv[1]; // User should input /home/ann/Desktop/Vitis-AI/model_zoo/vitis_ai_graduate_project/quantized_model/CNN_model6_int.xmodel
  	
    
    
    
	cout << "Model name: " << kernel_name << endl;
	cout << "IMAGE FOLDER PATH: " << path << endl;
  	cout << "OUTPUT file name: " << argv[3] << endl;
  	
	// join the path of all the images
	vector<cv::String> files;
	cv::glob(path, files);  // files[0] will be 'folder_name/aa.png'
	int count = files.size();
	/*
	for (int i = 0; i < count; i++) {
		cout << files[i] << " ";
	
	}
	cout << endl;
	*/
	cout << "The image count = " << count << endl;

	// Read image from a path.
  	vector<Mat> imgs;
  	vector<string> imgs_names;
  	for (int i = 0; i < count; i++) {
  	  // image file names.
  	  auto image = cv::imread(files[i]);
  	  if (image.empty()) {
 	     std::cerr << "Cannot load " << files[i] << std::endl;
 	     continue;
  	  }
  	  imgs.push_back(image);
  	  imgs_names.push_back(string(cv::String(files[i]).substr(length)));
  	}
 	if (imgs.empty()) {
 	   std::cerr << "No image load success!" << std::endl;
 	   return -1; // Exit with error code
	}
	cout << "Create a dpu task object." << endl;
	// Create a dpu task object.
 	auto task = vitis::ai::DpuTask::create(kernel_name);
 	if (!task) {
        std::cerr << "Failed to create DpuTask with model: " << kernel_name << std::endl;
        return -1; // Exit with error code
    }
 	auto batch = task->get_input_batch(0, 0);
 	cout << "batch " << batch << endl;
 	// Set the mean values and scale values.
 	task->setMeanScaleBGR({0.0f, 0.0f, 0.0f}, {0.00390625f, 0.00390625f, 0.00390625f});
 	// void setMeanScaleBGR(const std::vector< float > &mean, const std::vector< float > &scale)=0;
  	auto input_tensor = task->getInputTensor(0u);
  	cout << "input_tensor" << input_tensor[0] << endl;
  	CHECK_EQ((int)input_tensor.size(), 1) 
  		<< " the dpu model must have only one input";
  	// get the needed img size
  	auto width = input_tensor[0].width; 
  	auto height = input_tensor[0].height;
  	auto size = cv::Size(width, height); 
  	cout << "width " << width << endl;
  	cout << "height " << height << endl;
  	cout << "size " << size << endl;
  	
  	// Create a config and set the correlating data to control post-process.
  	vitis::ai::proto::DpuModelParam config;
  	// Fill all the parameters.
 	//auto ok = google::protobuf::TextFormat::ParseFromString(cnn_model6_int_config, &config);
  	//if (!ok) {
   	// 	cerr << "Set parameters failed!" << endl;
   	// 	abort();
  	//}
	
	vector<Mat> inputs;
 	vector<int> input_cols, input_rows;
  	for (long unsigned int i = 0, j = -1; i < imgs.size(); i++) {
		/* Pre-process Part */
		// (1) delete margin
		
		
		// (2) Resize it if its size is not match.
		cv::Mat image;
		input_cols.push_back(imgs[i].cols);
		input_rows.push_back(imgs[i].rows);
		cout << "Before resize " << imgs[i].size() << endl;
		if (size != imgs[i].size()) {
		  	cv::resize(imgs[i], image, size); // void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR );
		} else {
		  	image = imgs[i];
		}
		inputs.push_back(image);
		j++;
		if (j < batch - 1 && i < imgs.size() - 1) {
		  	continue;
		}
		cout << "inputs ::: " << inputs.size() << endl;
		cout << "Number of images: " << inputs.size() << endl;
		for (const auto& img : inputs) {
			cout << "Image size: " << img.size() << endl;
		}
		for (const auto& img : inputs) {
    		cout << "Image channels: " << img.channels() << endl;
		}
		// Set the input images into dpu.
		task->setImageRGB(inputs);

		/* DPU Runtime */
		// Run the dpu.
		task->run(0u);

		/* Post-process part */
		// Get output.
		auto output_tensors = task->getOutputTensor(0u);
		cout << i << ": Finish image=> " << imgs_names[i] << endl;
		cout << "Output tensor's size: " << output_tensors.size() << endl; // only 1 [batch_size, 
		for(int x = 0; x < output_tensors.size(); x++){
			cout << output_tensors[x] << " ";
			// format will be like :
			// output [ CNN_model6__CNN_model6_Sequential_class_layers__ModuleList_0__Linear_1__248_fix ] 
			// {, size=2, h=1, w=1, c=2, fixpos=4, virt= (0xffff9b55b000 )}
		}
		cout << endl;
		for (const auto& output_tensor : output_tensors) {
			//const float* output_data = reinterpret_cast<const float*>(output_tensor.get_data(0)); // get first batch
			const int32_t* raw_data = reinterpret_cast<const int32_t*>(output_tensor.get_data(0));
			
			
			// retrieves a pointer to the start of the data buffer for the first element (index 0) of the tensor.
			size_t output_size = (output_tensor.height * output_tensor.width * output_tensor.channel);
			
			std::vector<float> output_data(output_size);

			int fixpos = output_tensor.fixpos; // Number of fractional bits
			cout << "fixpos: " << fixpos << endl;
			for (size_t i = 0; i < output_size; ++i) {
				output_data[i] = raw_data[i] / static_cast<float>(1 << fixpos);
			}
			
			
			// do the softmax
			std::vector<float> probabilities = softmax(output_data);
			cout << "probabilities_size: " << probabilities.size() << endl;
			cout << "output_size: " << output_size << endl;
			//cout << "output_data buffer pointer: " << output_data << endl;
			
			out_fs << "Pic_" << i << "Image name:" << imgs_names[i] << " Raw data(int) = ";
			for (int x = 0; x < output_size; x++) {
				
				out_fs << x << ": " << raw_data[x] << " "; // use this pointer to access the data
			}
			out_fs << endl << endl;
			
			out_fs << "Pic_" << i << "Image name:" << imgs_names[i] << " Raw data(before softmax) = ";
			for (int x = 0; x < output_size; x++) {
				
				out_fs << x << ": " << output_data[x] << " "; // use this pointer to access the data
			}
			out_fs << endl << endl;
			
			out_fs << "Pic_" << i << "Image name:" << imgs_names[i] << " Result data = ";
			for (int x = 0; x < output_size; x++) {
				
				out_fs << x << ": " << probabilities[x] << " "; // use this pointer to access the data
			}
			out_fs << endl << endl;
		}
		
	
		/*auto output_tensors = task->getOutputTensor(0u);
		cout << i << ": Finish image=> " << imgs_names[i] << endl;
		//cout << "output_tensor.scores[0].score " << output_tensors.scores[0].score << endl;
		cout << "Output tensor's' size: " << output_tensors.size() << endl;
		for(int x = 0; x < output_tensors.size(); x++){
			cout << output_tensors[i] << " ";
		}
		cout << endl;
		for (const auto& output_tensor : output_tensors) {
			const float* output_data = reinterpret_cast<const float*>(output_tensor.get_data(0));
			size_t output_size = output_tensor.size / sizeof(float);
			cout << "output_size: " << output_size << endl;
			cout << "output_data: " << output_data << endl;
			//out_fs << "Output tensor size: [" << output_tensor.getDimensions()[0] << ", " << output_tensor.getDimensions()[1] << "]" << endl;
			cout << "Output tensor size: [" << output_tensor.height << ", " << output_tensor.width << "]" << endl;
			
			out_fs << "Output tensor data:" << endl;
			for (size_t i = 0; i < output_size; ++i) {
				out_fs << output_data[i] << " ";
			}
			out_fs << endl;
		}	
		*/
		//for (const auto& output_tensor : output_tensors) { // output_tensor => [batch_size, 2]
    	//	out_fs << "Output tensor size: [" << output_tensor.getDimensions()[0] << ", " << output_tensor.getDimensions()[1] << "]" << endl;
		//}
		//cout << "OUTPUT SIZE" << output_tensor.size() << endl;
		//for(int k = 0; k < (int)output_tensor.size(); k+=2){
		//	out_fs << imgs_names[k] << " " << output_tensor[k] << " " << output_tensor[k+1] << endl;
		//}
		inputs.clear();
		input_cols.clear();
		input_rows.clear();
		j = -1;
	}
       

	out_fs.close();
  	return 0;
}
