#include <cstdlib>
#include <iostream>
#include <string>
#include "wrapper.hpp"
#include "txtReader.hpp"

static long long parse_ll(const char* s) {
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    return v;
}

int main(int argc, char** argv){
	if (argc < 4) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0] << " <curve1.txt> <curve2.txt> [dim]\n"
            << "Example:\n"
            << "  " << argv[0] << " ../../OSM/000_463_000463672.txt ../../OSM/000_719_000719735.txt 2\n";
        return 1;
    }
	const std::string path1 = argv[1];
    const std::string path2 = argv[2];
	const int dim = static_cast<int>(parse_ll(argv[3]));
	bool forceOversize = false; 
	if(argc >= 5){
		if(argv[4] == std::string("1"))forceOversize = true;
	}
	int n = 1u<<30;
	printf("%d\n", forceOversize);
	if(dim == 2){
		Curve c1(txtLoader2(path1.c_str(), 0, n), dim);
		Curve c2(txtLoader2(path2.c_str(), 0, n), dim);
		if(c1.size() > c2.size())compute_distance_parallel(c2, c1, forceOversize);
		else compute_distance_parallel(c1, c2, forceOversize);
	}else if(dim == 3){
		Curve c1(txtLoader3(path1.c_str(), 0, n), dim);
		Curve c2(txtLoader3(path2.c_str(), 0, n), dim);
		if(c1.size() > c2.size())compute_distance_parallel(c2, c1, forceOversize);
		else compute_distance_parallel(c1, c2, forceOversize);
	}
		
	return 0;
}
