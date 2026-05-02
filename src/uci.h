#pragma once

#include <string>
#include <vector>
#include <sstream>

namespace Nexus {

class UCI {
public:
    static void loop();
    
    static std::vector<std::string> tokenize(const std::string& line);
    static void position(std::istringstream& is);
    static void go(std::istringstream& is);
    static void setoption(std::istringstream& is);
    
private:
    static void handle_command(const std::string& cmd);
};

} // namespace Nexus
