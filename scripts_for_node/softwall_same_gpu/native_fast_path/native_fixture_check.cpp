#include <openssl/evp.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Args {
    std::string fixture;
    std::string complex_sha;
    std::string p2p_sha;
    std::string tb_sha;
    std::size_t tb_bytes{};
};

bool is_little_endian() {
    const std::uint16_t value = 1;
    std::uint8_t first = 0;
    std::memcpy(&first, &value, sizeof(first));
    return first == 1;
}

std::string fixture_path(const Args& args, const std::string& name) {
    if (!args.fixture.empty() && args.fixture.back() == '/') {
        return args.fixture + name;
    }
    return args.fixture + "/" + name;
}

std::vector<std::uint8_t> read_all(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    const auto size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot size " + path);
    }
    std::vector<std::uint8_t> value(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(
        reinterpret_cast<char*>(value.data()),
        static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("cannot read " + path);
    }
    return value;
}

std::string sha256(const std::vector<std::uint8_t>& value) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr) {
        throw std::runtime_error("EVP_MD_CTX_new failed");
    }
    unsigned char digest[EVP_MAX_MD_SIZE]{};
    unsigned int digest_size = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1
        && EVP_DigestUpdate(context, value.data(), value.size()) == 1
        && EVP_DigestFinal_ex(context, digest, &digest_size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != 32) {
        throw std::runtime_error("SHA-256 failed");
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

std::uint32_t word(const std::vector<std::uint8_t>& bytes, std::size_t index) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
    return value;
}

Args parse(int argc, char** argv) {
    if (argc != 11) {
        throw std::runtime_error(
            "usage: checker --fixture DIR --complex-sha HEX --p2p-sha HEX "
            "--tb-sha HEX --tb-bytes N");
    }
    Args args;
    for (int index = 1; index < argc; index += 2) {
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--fixture") args.fixture = value;
        else if (key == "--complex-sha") args.complex_sha = value;
        else if (key == "--p2p-sha") args.p2p_sha = value;
        else if (key == "--tb-sha") args.tb_sha = value;
        else if (key == "--tb-bytes") args.tb_bytes = std::stoull(value);
        else throw std::runtime_error("unknown argument " + key);
    }
    return args;
}

int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);
        constexpr std::size_t n0 = 3276;
        constexpr std::size_t n1 = 14;
        constexpr std::size_t n2 = 4;
        constexpr std::size_t elements = n0 * n1 * n2;
        constexpr std::size_t raw_bytes = elements * 2 * sizeof(std::uint32_t);

        const auto complex = read_all(fixture_path(args, "rx_slot_complex64_fortran.bin"));
        const auto p2p = read_all(fixture_path(args, "rx_slot_real_imag_float32_c.bin"));
        const auto tb = read_all(fixture_path(args, "reference_tb_uint8.bin"));
        const bool little_endian = is_little_endian();
        const bool sizes = complex.size() == raw_bytes && p2p.size() == raw_bytes
            && tb.size() == args.tb_bytes;
        const bool hashes = sha256(complex) == args.complex_sha
            && sha256(p2p) == args.p2p_sha && sha256(tb) == args.tb_sha;

        bool layout = little_endian && sizes;
        if (layout) {
            for (std::size_t first = 0; first < n0 && layout; ++first) {
                for (std::size_t second = 0; second < n1 && layout; ++second) {
                    for (std::size_t third = 0; third < n2; ++third) {
                        const std::size_t f_index = first + n0 * (second + n1 * third);
                        const std::size_t c_index = (first * n1 + second) * n2 + third;
                        const bool same_real = word(complex, 2 * f_index) == word(p2p, c_index);
                        const bool same_imag = word(complex, 2 * f_index + 1)
                            == word(p2p, elements + c_index);
                        if (!same_real || !same_imag) {
                            layout = false;
                            break;
                        }
                    }
                }
            }
        }
        const bool all_pass = little_endian && sizes && hashes && layout;
        std::cout << "{\n"
                  << "  \"schema\": \"softwall-c166-native-cpp-fixture-check-v1\",\n"
                  << "  \"little_endian\": " << (little_endian ? "true" : "false") << ",\n"
                  << "  \"sizes_match\": " << (sizes ? "true" : "false") << ",\n"
                  << "  \"sha256_match\": " << (hashes ? "true" : "false") << ",\n"
                  << "  \"logical_layout_match\": " << (layout ? "true" : "false") << ",\n"
                  << "  \"all_pass\": " << (all_pass ? "true" : "false") << "\n"
                  << "}\n";
        return all_pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
