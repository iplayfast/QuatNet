#include "GGUFHelper.hpp"
#include <iostream>
#include <cstring>

namespace Quaternary {

struct GGUFHeader {
    uint32_t magic = 0x46554747; // "GGUF"
    uint32_t version = 3;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

bool GGUFHelper::save(const std::string& filename, const LifetimeNN& model) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    GGUFHeader header;
    header.tensor_count = model.heads.size();
    header.metadata_kv_count = 7; // arch, v_size, h_count, q_version, alignment, running_mse, bias
    out.write(reinterpret_cast<char*>(&header), sizeof(header));

    write_kv(out, "general.architecture", "quaternary_nn");
    write_kv_uint32(out, "general.vector_size", model.vector_size);
    write_kv_uint32(out, "general.head_count", model.num_heads);
    write_kv_uint32(out, "general.quantization_version", 1);
    write_kv_uint32(out, "general.alignment", 32);
    write_kv_float(out, "general.running_mse", model.running_mse);
    write_kv_float(out, "general.bias", model.bias);

    // Calculate dynamic offsets
    uint64_t current_offset = 0;
    size_t bytes_per_tensor = model.vector_size / 4;
    if (model.vector_size % 4 != 0) bytes_per_tensor++;
    
    // We need to account for alignment padding before the first tensor
    // This is tricky because we don't know the exact size of the tensor info block yet.
    // In GGUF, offsets are relative to the start of the data blob.
    
    // Tensor info block
    for (int i = 0; i < model.num_heads; ++i) {
        write_string(out, "head." + std::to_string(i) + ".weight");
        uint32_t n_dims = 1;
        out.write(reinterpret_cast<char*>(&n_dims), 4);
        uint64_t dim = model.vector_size;
        out.write(reinterpret_cast<char*>(&dim), 8);
        uint32_t type = GGML_TYPE_Q2_Q;
        out.write(reinterpret_cast<char*>(&type), 4);
        
        out.write(reinterpret_cast<char*>(&current_offset), 8);
        
        // Update offset for next tensor, ensuring 32-byte alignment
        uint64_t tensor_size = bytes_per_tensor;
        uint64_t padded_size = (tensor_size + 31) & ~31;
        current_offset += padded_size;
    }

    align_file(out, 32);

    // Tensor data (packed)
    for (const auto& head : model.heads) {
        size_t bytes_written = 0;
        for (size_t i = 0; i < head.size(); i += 4) {
            uint8_t packed = 0;
            auto get_bits = [&](float w) -> uint8_t {
                return model.to_bits(model.quantize(w));
            };
            packed |= (get_bits(head[i]) << 6);
            if (i+1 < head.size()) packed |= (get_bits(head[i+1]) << 4);
            if (i+2 < head.size()) packed |= (get_bits(head[i+2]) << 2);
            if (i+3 < head.size()) packed |= get_bits(head[i+3]);
            out.write(reinterpret_cast<char*>(&packed), 1);
            bytes_written++;
        }
        // Pad individual tensor data to 32 bytes
        int padding = (32 - (bytes_written % 32)) % 32;
        for (int p = 0; p < padding; ++p) out.put(0);
    }

    return true;
}

bool GGUFHelper::load(const std::string& filename, LifetimeNN& model) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    GGUFHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.magic != 0x46554747) return false;

    int v_size = 0;
    int h_count = 0;

    for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
        std::string key = read_string(in);
        uint32_t type;
        in.read(reinterpret_cast<char*>(&type), 4);
        if (type == 8) { // String
            read_string(in); 
        } else if (type == 4) { // Uint32
            uint32_t val;
            in.read(reinterpret_cast<char*>(&val), 4);
            if (key == "general.vector_size") v_size = val;
            if (key == "general.head_count") h_count = val;
        } else if (type == 7) { // FLOAT32
            float val;
            in.read(reinterpret_cast<char*>(&val), 4);
            if (key == "general.running_mse") model.running_mse = val;
            if (key == "general.bias") model.bias = val;
        }
    }

    if (v_size == 0 || h_count == 0) return false;

    // Skip tensor info
    for (uint64_t i = 0; i < header.tensor_count; ++i) {
        read_string(in);
        uint32_t n_dims;
        in.read(reinterpret_cast<char*>(&n_dims), 4);
        in.seekg(n_dims * 8, std::ios::cur); // Skip dims
        in.seekg(4 + 8, std::ios::cur); // Skip type and offset
    }

    // Align to 32
    long pos = in.tellg();
    int padding = (32 - (pos % 32)) % 32;
    in.seekg(padding, std::ios::cur);

    std::vector<std::vector<float>> heads;
    for (int h = 0; h < h_count; ++h) {
        std::vector<float> head(v_size);
        size_t bytes_read = 0;
        for (int i = 0; i < v_size; i += 4) {
            uint8_t packed;
            in.read(reinterpret_cast<char*>(&packed), 1);
            head[i]   = model.from_bits((packed >> 6) & 0b11);
            if (i+1 < v_size) head[i+1] = model.from_bits((packed >> 4) & 0b11);
            if (i+2 < v_size) head[i+2] = model.from_bits((packed >> 2) & 0b11);
            if (i+3 < v_size) head[i+3] = model.from_bits(packed & 0b11);
            bytes_read++;
        }
        // Skip padding to next 32-byte boundary
        int padding = (32 - (bytes_read % 32)) % 32;
        in.seekg(padding, std::ios::cur);
        
        heads.push_back(head);
    }

    model.load_from_raw(v_size, heads);
    return true;
}

void GGUFHelper::write_string(std::ofstream& out, const std::string& s) {
    uint64_t len = s.length();
    out.write(reinterpret_cast<char*>(&len), 8);
    out.write(s.c_str(), len);
}

void GGUFHelper::write_kv(std::ofstream& out, const std::string& key, const std::string& val) {
    write_string(out, key);
    uint32_t type = 8; // GGUF_TYPE_STRING
    out.write(reinterpret_cast<char*>(&type), 4);
    write_string(out, val);
}

void GGUFHelper::write_kv_uint32(std::ofstream& out, const std::string& key, uint32_t val) {
    write_string(out, key);
    uint32_t type = 4; // GGUF_TYPE_UINT32
    out.write(reinterpret_cast<char*>(&type), 4);
    out.write(reinterpret_cast<char*>(&val), 4);
}
void GGUFHelper::write_kv_float(std::ofstream& out, const std::string& key, float val) {
    write_string(out, key);
    uint32_t type = 7; // GGUF_TYPE_FLOAT32 is type 7
    out.write(reinterpret_cast<char*>(&type), 4);
    out.write(reinterpret_cast<char*>(&val), 4);
}
void GGUFHelper::align_file(std::ofstream& out, int alignment) {
    long pos = out.tellp();
    int padding = (alignment - (pos % alignment)) % alignment;
    for (int i = 0; i < padding; ++i) out.put(0);
}

std::string GGUFHelper::read_string(std::ifstream& in) {
    uint64_t len;
    in.read(reinterpret_cast<char*>(&len), 8);
    std::string s(len, '\0');
    in.read(&s[0], len);
    return s;
}

} // namespace Quaternary
