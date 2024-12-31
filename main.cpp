#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <cmath>

//#define SHOW_IMAGE

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

// Helper function to create a color wheel based on 
// https://github.com/tomrunia/OpticalFlow_Visualization
cv::Mat makeColorwheel() {
    const int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
    const int ncols = RY + YG + GC + CB + BM + MR;
    cv::Mat colorwheel(ncols, 1, CV_8UC3);

    int col = 0;
    // RY
    for (int i = 0; i < RY; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 255 * i / RY, 0);
    }
    // YG
    for (int i = 0; i < YG; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 - 255 * i / YG, 255, 0);
    }
    // GC
    for (int i = 0; i < GC; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255, 255 * i / GC);
    }
    // CB
    for (int i = 0; i < CB; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255 - 255 * i / CB, 255);
    }
    // BM
    for (int i = 0; i < BM; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 * i / BM, 0, 255);
    }
    // MR
    for (int i = 0; i < MR; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 0, 255 - 255 * i / MR);
    }
    return colorwheel;
}

// Helper function to visualize optical flow based on 
// https://github.com/tomrunia/OpticalFlow_Visualization
cv::Mat visualizeFlow(const at::Tensor& flow) {
    auto flow_cpu = flow.squeeze().permute({1, 2, 0}).contiguous().detach().cpu();
    cv::Mat flow_mat(flow_cpu.size(0), flow_cpu.size(1), CV_32FC2);
    std::memcpy(flow_mat.data, flow_cpu.data_ptr<float>(), 
                flow_cpu.numel() * sizeof(float));

    cv::Mat flow_parts[2];
    cv::split(flow_mat, flow_parts);
    cv::Mat u = flow_parts[0], v = flow_parts[1];

    cv::Mat magnitude, angle;
    cv::cartToPolar(u, v, magnitude, angle);

    // Normalize magnitude
    double mag_max;
    cv::minMaxLoc(magnitude, 0, &mag_max);
    if (mag_max > 0) {
        magnitude /= mag_max;
    }

    // Convert angle to [0, 1] range
    angle *= (1.0 / (2 * CV_PI));
    angle += 0.5;

    // Apply color wheel
    cv::Mat colorwheel = makeColorwheel();
    const int ncols = colorwheel.rows;
    cv::Mat flow_color(flow_mat.size(), CV_8UC3);

    for (int i = 0; i < flow_mat.rows; ++i) {
        for (int j = 0; j < flow_mat.cols; ++j) {
            float mag = magnitude.at<float>(i, j);
            float ang = angle.at<float>(i, j);

            int k0 = static_cast<int>(ang * (ncols - 1));
            int k1 = (k0 + 1) % ncols;
            float f = (ang * (ncols - 1)) - k0;

            cv::Vec3b col0 = colorwheel.at<cv::Vec3b>(k0);
            cv::Vec3b col1 = colorwheel.at<cv::Vec3b>(k1);

            cv::Vec3b color;
            for (int ch = 0; ch < 3; ++ch) {
                float col = (1 - f) * col0[ch] + f * col1[ch];
                if (mag <= 1) {
                    col = 255 - mag * (255 - col);
                } else {
                    col *= 0.75;
                }
                color[ch] = static_cast<uchar>(col);
            }

            flow_color.at<cv::Vec3b>(i, j) = color;
        }
    }

    return flow_color;
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

    // Set up video writer for side-by-side output
    cv::VideoWriter video_out("output.avi", 
                              cv::VideoWriter::fourcc('M','J','P','G'), 
                              fps, 
                              cv::Size(frame_width * 2, frame_height));  // Double the width for side-by-side
    // Process video
    cv::Mat frame, prev_frame, output_frame, display_frame;

    // Set model input dimensions
    int model_input_width = 960, model_input_height = 520; // Updated to match Python example
    
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
            
            // Create side-by-side display
            display_frame = cv::Mat(frame_height, frame_width * 2, CV_8UC3);
            frame.copyTo(display_frame(cv::Rect(0, 0, frame_width, frame_height)));
            output_frame.copyTo(display_frame(cv::Rect(frame_width, 0, frame_width, frame_height)));
            // reconvert back display_frame to BGR for imshow
            cv::cvtColor(display_frame, display_frame, cv::COLOR_RGB2BGR);
#ifdef SHOW_IMAGE
            cv::imshow("Original vs Optical Flow", display_frame);
            // Break on 'q' key press
            if (cv::waitKey(delay) == 'q') break;        
#endif                
            video_out.write(display_frame);

            // Update previous frame
            prev_frame = frame.clone();



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