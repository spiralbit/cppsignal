#include "generate.hpp"
#include "filter.hpp"
#include <iostream>
#include <string_view>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: cps <generate|filter> [options]\n";
        return 1;
    }

    std::string_view cmd{argv[1]};

    try {
        if (cmd == "generate") {
            auto opts = cps::cli::parse_generate_args(argc, argv);
            cps::cli::generate(std::cout, opts);
        } else if (cmd == "filter") {
            auto opts = cps::cli::parse_filter_args(argc, argv);
            cps::cli::filter_stream(std::cin, std::cout, opts);
        } else {
            std::cerr << "cps: unknown command '" << cmd << "'\n";
            return 1;
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "cps: " << e.what() << '\n';
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "cps: error: " << e.what() << '\n';
        return 1;
    }
}
