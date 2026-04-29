#pragma once

#include "QuaternaryNN.hpp"
#include <string>
#include <vector>
#include <fstream>

namespace Quaternary {

#define GGML_TYPE_Q2_F 99
#define GGML_TYPE_Q2_Q GGML_TYPE_Q2_F  // quaternary (4-value) quantization

class GGUFHelper {
public:
    static bool save(const std::string& filename, const LifetimeNN& model);
    static bool load(const std::string& filename, LifetimeNN& model);

private:
    static void write_string(std::ofstream& out, const std::string& s);
    static void write_kv(std::ofstream& out, const std::string& key, const std::string& val);
    static void write_kv_uint32(std::ofstream& out, const std::string& key, uint32_t val);
    static void align_file(std::ofstream& out, int alignment);
    static void write_kv_float(std::ofstream& out, const std::string& key, float val);
    
    static std::string read_string(std::ifstream& in);
};

} // namespace Quaternary
