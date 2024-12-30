#include <opencv2/opencv.hpp>
#include <torch/script.h> // LibTorch
#include <iostream>
#include <memory>

// Helper function to preprocess an OpenCV frame for the model
torch::Tensor preprocessFrame(const cv::Mat& frame, int height, int width) {
    cv::Mat resized, float_img;
    cv::resize(frame, resized, cv::Size(width, height));
    resized.convertTo(float_img, CV_32F, 1.0 / 255); // Normalize to [0, 1]
    torch::Tensor tensor = torch::from_blob(float_img.data, {1, height, width, 3}, torch::kFloat);
    tensor = tensor.permute({0, 3, 1, 2}); // Convert to [B, C, H, W]
    return tensor.clone(); // Clone to ensure memory ownership
}

// Helper function to visualize optical flow (placeholder, modify as needed)
cv::Mat visualizeFlow(const at::Tensor& output) {
    // Convert the PyTorch tensor to OpenCV Mat
    auto flow = output.squeeze(0).permute({1, 2, 0}).detach().cpu(); // [H, W, 2]
    cv::Mat flow_mat(flow.size(0), flow.size(1), CV_32FC2, flow.data_ptr<float>());

    // Split flow into horizontal (x) and vertical (y) components
    cv::Mat flow_parts[2];
    cv::split(flow_mat, flow_parts); // flow_parts[0] = horizontal, flow_parts[1] = vertical

    // Compute magnitude and angle
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

    // Normalize magnitude to range [0, 1]
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    // Scale angle to [0, 1] for HSV hue channel
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    // Build HSV image
    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;                                // Hue (angle)
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);  // Saturation (set to 1)
    _hsv[2] = magn_norm;                            // Value (normalized magnitude)
    cv::merge(_hsv, 3, hsv);                        // Merge HSV channels into one image

    // Convert HSV to 8-bit and then to BGR for visualization
    hsv.convertTo(hsv8, CV_8U, 255.0); // Scale to 8-bit
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR); // Convert HSV to BGR for visualization

    return bgr; // Return the final BGR image for display
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> <model_path>\n";
        return -1;
    }

    std::string video_path = argv[1];
    std::string model_path = argv[2];


    // Open video file
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }

    // Load model
    torch::jit::Module torch_model;


    try {

        torch_model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading TorchScript model: " << e.what() << "\n";
        return -1;
    }

    // Process video
    cv::Mat frame, prev_frame, output_frame;
    int frame_width = 256, frame_height = 256; // Resize for model input
    while (cap.read(frame)) 
    {
        // Convert current frame to RGB
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Handle the first frame
        if (prev_frame.empty()) {
            prev_frame = frame.clone(); // Clone the first frame
            cv::resize(prev_frame, prev_frame, cv::Size(frame_width, frame_height)); // Ensure proper size
            continue;
        }

        // Preprocess frames for the model
        auto input1 = preprocessFrame(prev_frame, frame_height, frame_width);
        auto input2 = preprocessFrame(frame, frame_height, frame_width);

        torch::Tensor output;
        try {
            // TorchScript inference
            std::vector<torch::jit::IValue> inputs = {input1, input2};
            auto result = torch_model.forward(inputs);
            if (result.isTensor()) {
                output = result.toTensor();
            } else if (result.isList()) {
                auto list = result.toList();
                // Extract the last tensor from the list (most refined optical flow)
                output = list.get(list.size() - 1).toTensor(); // Use clone to ensure alignment
            } else {
                throw std::runtime_error("Unsupported output type from the model");
            }


            // Visualize the output
            output_frame = visualizeFlow(output);
            cv::imshow("Optical Flow", output_frame);

            // Update previous frame
            prev_frame = frame.clone();
            cv::resize(prev_frame, prev_frame, cv::Size(frame_width, frame_height)); // Resize to match input size

            // Break on key press
            if (cv::waitKey(30) >= 0) break;

        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
