#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <chrono>

// Helper function to preprocess an OpenCV frame for the model
torch::Tensor preprocessFrame(const cv::Mat& frame, int height, int width) {
    cv::Mat resized, float_img;
    cv::resize(frame, resized, cv::Size(width, height));
    resized.convertTo(float_img, CV_32F, 1.0 / 255); // Normalize to [0, 1]
    float_img = (float_img - 0.5) / 0.5; // Normalize to [-1, 1]
    torch::Tensor tensor = torch::from_blob(float_img.data, {1, height, width, 3}, torch::kFloat);
    tensor = tensor.permute({0, 3, 1, 2}); // Convert to [B, C, H, W]
    return tensor.clone(); // Clone to ensure memory ownership
}

// Helper function to visualize optical flow
cv::Mat visualizeFlow(const at::Tensor& flow) {
    auto flow_cpu = flow.squeeze().permute({1, 2, 0}).detach().cpu();
    cv::Mat flow_mat(flow_cpu.size(0), flow_cpu.size(1), CV_32FC2);
    std::memcpy(flow_mat.data, flow_cpu.data_ptr<float>(), 
                flow_cpu.numel() * sizeof(float));

    cv::Mat magnitude, angle, hsv;
    cv::cartToPolar(cv::Mat(flow_mat.size(), CV_32F, flow_mat.ptr<float>(0)),
                    cv::Mat(flow_mat.size(), CV_32F, flow_mat.ptr<float>(1)),
                    magnitude, angle);

    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

    // Create HSV image
    hsv = cv::Mat(flow_mat.size(), CV_8UC3);
    for(int i = 0; i < flow_mat.rows; i++) {
        for(int j = 0; j < flow_mat.cols; j++) {
            hsv.at<cv::Vec3b>(i,j) = cv::Vec3b(
                cv::saturate_cast<uchar>(angle.at<float>(i,j) * 180 / CV_PI / 2),
                255,
                cv::saturate_cast<uchar>(magnitude.at<float>(i,j) * 255)
            );
        }
    }

    cv::Mat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
    return rgb;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_path> <model_path>\n";
        return -1;
    }

    std::string video_path = argv[1];
    std::string model_path = argv[2];
    torch::Device device(torch::kCPU);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << "\n";
        return -1;
    }


    // Load model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        model.to(device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading TorchScript model: " << e.what() << "\n";
        return -1;
    }

    // Get video properties
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = 1000 / fps;

    // Set up video writer
    cv::VideoWriter video_out("output.avi", 
                              cv::VideoWriter::fourcc('M','J','P','G'), 
                              fps, 
                              cv::Size(frame_width, frame_height));

    // Process video
    cv::Mat frame, prev_frame, output_frame;
    int model_input_width = 256, model_input_height = 256; // Resize for model input
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (cap.read(frame)) {
        frame_count++;

        // Convert current frame to RGB
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Handle the first frame
        if (prev_frame.empty()) {
            prev_frame = frame.clone();
            continue;
        }

        // Preprocess frames for the model
        auto input1 = preprocessFrame(prev_frame, model_input_height, model_input_width).to(device);
        auto input2 = preprocessFrame(frame, model_input_height, model_input_width).to(device);

        torch::Tensor output;
        try {
            // TorchScript inference
            torch::NoGradGuard no_grad;
            std::vector<torch::jit::IValue> inputs = {input1, input2};
            auto result = model.forward(inputs);
            
            if (result.isTensor()) {
                output = result.toTensor();
            } else if (result.isList()) {
                auto list = result.toListRef();  // Changed from result.toList().elements()
                output = list.back().toTensor();
            } else {
                throw std::runtime_error("Unsupported output type from the model");
            }

            // Visualize the output
            output_frame = visualizeFlow(output);
            
            // Resize output_frame to match original frame size
            cv::resize(output_frame, output_frame, cv::Size(frame_width, frame_height));
            
            cv::imshow("Optical Flow", output_frame);
            video_out.write(output_frame);

            // Update previous frame
            prev_frame = frame.clone();

            // Break on 'q' key press
            if (cv::waitKey(delay) == 'q') break;

        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Processed " << frame_count << " frames in " << duration.count() / 1000.0 << " seconds" << std::endl;
    std::cout << "Average FPS: " << frame_count / (duration.count() / 1000.0) << std::endl;

    cap.release();
    video_out.release();
    cv::destroyAllWindows();
    return 0;
}