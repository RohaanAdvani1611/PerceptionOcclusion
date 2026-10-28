#include "core/Engine.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
    EngineOptions opts;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench") && i + 1 < argc) {
            opts.benchFrames = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
            opts.csvPath = argv[++i];
        } else {
            std::cerr << "Usage: " << argv[0] << " [--bench N] [--csv path]\n";
            return 1;
        }
    }

    Engine engine(opts);
    engine.run();
    return 0;
}
