#include <fstream>
#include <vector>

#include "geometry_basics.hpp"

Points txtLoader2(const std::string& path, int START=0, int LIMIT=10000000){
	std::string line;
	Points ret;
	
	std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return ret;
    }

	int count = 0;
	
	while (std::getline(fin, line)) {
		if (count >= LIMIT) break;
		
        std::stringstream ss(line);
        std::string token;
		Point pt;
        while (std::getline(ss, token, ',')) {
            try {
                double val = std::stod(token);
                pt.coordinates.push_back(val);
            } catch (const std::exception& e) {
                std::cerr << "Warning: invalid double in line: " << line << std::endl;
            }
        }
		if (!pt.coordinates.empty()) {
			count ++;
			if(count < START) continue;
			if(!ret.empty()){
				auto last = ret.back();
				if(last.coordinates[0] != pt.coordinates[0] || last.coordinates[1]!=pt.coordinates[1]){
					ret.push_back(std::move(pt));
				}
			}else{ret.push_back(std::move(pt));}
		}
    }

	fin.close();
	return ret;
}

Points txtLoader3(const std::string& path, int START=0, int LIMIT=10000000){
	std::string line;
	Points ret;
	
	std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return ret;
    }

	int count = 0;
	
	while (std::getline(fin, line)) {
		if (count >= LIMIT) break;
		
        std::stringstream ss(line);
        std::string token;
		Point pt;
        while (std::getline(ss, token, ',')) {
            try {
                double val = std::stod(token);
                pt.coordinates.push_back(val);
            } catch (const std::exception& e) {
                std::cerr << "Warning: invalid double in line: " << line << std::endl;
            }
        }
		if (!pt.coordinates.empty()) {
			count ++;
			if(count < START) continue;
			if(!ret.empty()){
				auto last = ret.back();
				if(last.coordinates[0] != pt.coordinates[0] || last.coordinates[1]!=pt.coordinates[1] || last.coordinates[2] != pt.coordinates[2]){
					ret.push_back(std::move(pt));
				}
			}else{ret.push_back(std::move(pt));}
		}
    }

	fin.close();
	return ret;
}
