#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "QuaternaryNN.hpp"
#include "GGUFHelper.hpp"

using namespace Quaternary;

void print_usage() {
    std::cout << "Usage: quaternary_nn <mode> <model_file> [args...]" << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  train  <model_file> <lr>  - Read x and target from stdin" << std::endl;
    std::cout << "  infer  <model_file> <x>   - Print prediction for x" << std::endl;
    std::cout << "  export <model_file>       - Dump weights to stdout for GGUF conversion" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string mode = argv[1];
    std::string model_path = argv[2];
    
    // Fix 1: Match your constructor which only takes initial_size
    LifetimeNN model(1024); 

    if (GGUFHelper::load(model_path, model)) {
        std::cerr << "[INIT] Loaded existing model: " << model_path << std::endl;
    } else {
        std::cerr << "[INIT] Creating new model: " << model_path << std::endl;
    }

    if (mode == "train") {
        if (argc < 4) {
            std::cerr << "Error: Training requires a learning rate." << std::endl;
            return 1;
        }
        float lr = std::stof(argv[3]);
        std::cerr << "[TRAIN] Starting training loop (LR: " << lr << ")..." << std::endl;

        float x, target;
        while (std::cin >> x >> target) {
            // Get prediction BEFORE training
            float prediction = model.forward(x);
            
            // Output for Python to capture
            std::cout << "PRED: " << prediction << std::endl << std::flush;
            
            // Update the model
            model.train_step(x, target, lr);

            // Fix 2: Remove the direct plateau_ticks check as your header 
            // likely handles growth logic inside train_step
        }
        
        if (GGUFHelper::save(model_path, model)) {
            std::cerr << "[TRAIN] Training complete. Model saved." << std::endl;
        }

    } else if (mode == "infer") {
        if (argc < 4) {
            std::cerr << "Error: Inference requires an input value x." << std::endl;
            return 1;
        }
        float x = std::stof(argv[3]);
        std::cout << model.forward(x) << std::endl;
    } else if (mode == "export") {
        // Export: dump model state to stdout for gguf-py
        // Format:
        //   VECTOR_SIZE=<n>
        //   NUM_HEADS=<n>
        //   BIAS=<float>
        //   RUNNING_MSE=<float>
        //   HEAD_WEIGHTS <head_idx> <w0> <w1> ... <wn>
        std::cout << "VECTOR_SIZE=" << model.vector_size << std::endl;
        std::cout << "NUM_HEADS=" << model.num_heads << std::endl;
        std::cout << "BIAS=" << model.bias << std::endl;
        std::cout << "RUNNING_MSE=" << model.running_mse << std::endl;
        for (int h = 0; h < model.num_heads; h++) {
            std::cout << "HEAD_WEIGHTS " << h;
            for (float w : model.heads[h]) {
                std::cout << " " << w;
            }
            std::cout << std::endl;
        }
        std::cerr << "[EXPORT] Dumped model state" << std::endl;
    } else {
        print_usage();
        return 1;
    }

    return 0;
}
