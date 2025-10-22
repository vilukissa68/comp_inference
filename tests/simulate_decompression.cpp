#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include "wrapper.hpp"


std::vector<uint8_t> generate_array_uint8(size_t size) {
    std::vector<uint8_t> data(size);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 127);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(dist(rng));
    }
    return data;
}

bool test_simulate_decompression() {
    constexpr double GIGS = 5.0;
    const size_t SIZE = static_cast<size_t>(GIGS * 1024.0 * 1024.0 * 1024.0); // bytes
    constexpr double R = 1.00;

    const size_t size_reduced = static_cast<size_t>(SIZE * R);
    std::vector<uint8_t> output_data(SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    auto input_data = generate_array_uint8(size_reduced);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Data generation time: " << duration.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();

    simulate_decompression(
        input_data.data(),         // pointer to input
        output_data.data(),        // pointer to output
        input_data.size(),         // input size
        output_data.size()         // output size
    );

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;

    std::cout << "Decompression simulation time: " << duration.count() << " seconds\n";

    // Print sample data
    std::cout << "First 10 elements of input data: ";
    for (size_t i = 0; i < 10 && i < input_data.size(); ++i)
        std::cout << static_cast<int>(input_data[i]) << " ";
    std::cout << "\n";

    std::cout << "Last 10 elements of input data: ";
    for (size_t i = input_data.size() > 10 ? input_data.size() - 10 : 0; i < input_data.size(); ++i)
        std::cout << static_cast<int>(input_data[i]) << " ";
    std::cout << "\n";

    std::cout << "First 10 elements of output data: ";
    for (size_t i = 0; i < 10 && i < output_data.size(); ++i)
        std::cout << static_cast<int>(output_data[i]) << " ";
    std::cout << "\n";

    std::cout << "Last 10 elements of output data: ";
    for (size_t i = output_data.size() > 10 ? output_data.size() - 10 : 0; i < output_data.size(); ++i)
        std::cout << static_cast<int>(output_data[i]) << " ";
    std::cout << "\n";

    std::cout << "Output data size: " << output_data.size() << "\n";
    return true;
}

int main() {
    if (test_simulate_decompression()) {
        std::cout << "test_simulate_decompression completed successfully.\n";
    }
    return 0;
}

