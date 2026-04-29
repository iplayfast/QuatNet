#include "QuaternaryNN.hpp"
#include "GGUFHelper.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>" << std::endl;
        return 1;
    }
    
    Quaternary::LifetimeNN model;
    std::string model_path = argv[1];
    
    if (Quaternary::GGUFHelper::load(model_path, model)) {
        std::cerr << "[SAVE] Loaded model, saving..." << std::endl;
        if (Quaternary::GGUFHelper::save(model_path, model)) {
            std::cerr << "[SAVE] Done. Model saved to " << model_path << std::endl;
        } else {
            std::cerr << "[SAVE] Failed!" << std::endl;
            return 1;
        }
    } else {
        std::cerr << "[SAVE] Could not load model!" << std::endl;
        return 1;
    }
    
    return 0;
}