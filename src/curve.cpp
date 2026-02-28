#include "curve.hpp"

Curve::Curve(const Points& points, unsigned int dimensions) : 
	points(points), number_dimensions(dimensions) {
	if (points.empty()) { 
		std::cerr << "warning: constructed empty curve" << std::endl;
		return; 
	}

	#if DEBUG
	std::cout << "constructed curve of complexity " << points.size() << std::endl;
	#endif
}

void Curve::push_back(Point const& point)
{
	points.push_back(point);
}

