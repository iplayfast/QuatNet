#pragma once

#include "Q2_ValueSet.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace Quaternary {

class LifetimeNN {
public:
    int vector_size;
    int num_heads;
    std::vector<std::vector<float>> heads; 
    float running_mse = 0.0f;
    float last_mse = 0.0f;
    float bias = 0.0f;
    int plateau_patience = 0;
    static constexpr int MAX_VECTOR_SIZE = 16384;

    // Active value set (from Q2_ValueSet.hpp)
    static constexpr ValueSet VS = ACTIVE_VALUES;

    LifetimeNN(int initial_size = 1024);

    float quantize(float w) const;
    uint8_t to_bits(float q) const;
    float from_bits(uint8_t bits) const;

    float forward(float input) const;
    void train_step(float input, float target, float lr);
    void expand();

    void load_from_raw(int v_size, const std::vector<std::vector<float>>& data);
};

} // namespace Quaternary
