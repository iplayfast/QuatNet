#include "QuaternaryNN.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>

namespace Quaternary {

LifetimeNN::LifetimeNN(int initial_size) 
    : vector_size(initial_size), num_heads(1) {
    std::vector<float> initial_head(vector_size);
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

    for (int i = 0; i < vector_size; ++i) {
        float base = (i % 2 == 0) ? 0.02f : -0.02f;
        initial_head[i] = base + dist(gen);
    }
    heads.push_back(initial_head);
}

float LifetimeNN::quantize(float w) const {
    if (w > VS.threshold_hi) return VS.pos_strong;
    if (w > 0.0f)            return VS.pos_weak;
    if (w > VS.threshold_lo) return VS.neg_weak;
    return VS.neg_strong;
}

uint8_t LifetimeNN::to_bits(float q) const {
    if (q == VS.pos_strong) return 0b00;
    if (q == VS.pos_weak)   return 0b01;
    if (q == VS.neg_weak)   return 0b10;
    return 0b11;
}

float LifetimeNN::from_bits(uint8_t bits) const {
    if (bits == 0b00) return VS.pos_strong;
    if (bits == 0b01) return VS.pos_weak;
    if (bits == 0b10) return VS.neg_weak;
    return VS.neg_strong;
}

float LifetimeNN::forward(float input) const {
    float total = 0;
    for (const auto& head : heads) {
        for (float w : head) {
            total += input * quantize(w);
        }
    }
    return total + bias;
}

void LifetimeNN::train_step(float input, float target, float lr) {
    float prediction = forward(input);
    float error = prediction - target;

    if (std::isnan(error)) {
        std::cerr << "[WARN] NaN detected! Reducing LR may help." << std::endl;
        return;
    }

    float prev_mse = running_mse;
    running_mse = (0.999f * running_mse) + (0.001f * (error * error));

    for (auto& head : heads) {
        for (float& w : head) {
            w -= lr * error * input;
        }
    }
    bias -= lr * error;

    static int plateau_ticks = 0;
    if (std::abs(running_mse - prev_mse) < 0.0001f) { 
        plateau_ticks++;
        if (plateau_ticks > 5000) { 
            expand();
            plateau_ticks = 0;
        }
    } else {
        plateau_ticks = 0;
    }
}

void LifetimeNN::expand() {
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<float> dist(-0.001f, 0.001f);

    if (vector_size < MAX_VECTOR_SIZE) {
        std::cerr << "[GROWTH] Widening vector: " << vector_size << " -> " << vector_size * 2 << std::endl;
        int old_size = vector_size;
        vector_size *= 2;
        for (auto& h : heads) {
            h.resize(vector_size);
            for (int i = old_size; i < vector_size; ++i) {
                h[i] = dist(gen);
            }
        }
    } else {
        std::cerr << "[GROWTH] Adding new Head: " << num_heads + 1 << std::endl;
        std::vector<float> new_head(vector_size);
        for (int i = 0; i < vector_size; ++i) {
            new_head[i] = dist(gen);
        }
        heads.push_back(new_head);
        num_heads++;
    }
}

void LifetimeNN::load_from_raw(int v_size, const std::vector<std::vector<float>>& data) {
    vector_size = v_size;
    heads = data;
    num_heads = data.size();
    running_mse = 0.0f;
}

} // namespace Quaternary
