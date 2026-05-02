#include "uci.h"
#include "tt.h"

int main(int argc, char* argv[]) {
    // Initialize TT with default 64MB
    Nexus::TT.resize(64);
    
    // Run UCI loop
    Nexus::UCI::loop();
    
    return 0;
}
