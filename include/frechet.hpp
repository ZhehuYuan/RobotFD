#pragma once

#include <boost/chrono/include.hpp>

#include "geometry_basics.hpp"
#include "curve.hpp"
#include "intersection_algorithm_in_parallel.hpp"

namespace Frechet {
namespace Continuous {
	
	auto distance(const Curve&, const Curve&, double, double, 
			const double = std::numeric_limits<double>::epsilon(), bool = true) -> double;
	double distance_cuda(const Curve&, const Curve&, const double eps, bool forceOversize);
	bool _lessThan(const double, const Curve&, const Curve&, 
			double*, double*);
}
}
