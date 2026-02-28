#include <cfloat>
#include "wrapper.hpp"

namespace fc = Frechet::Continuous;

const double eps = FLT_EPSILON;

double compute_distance_parallel(const Curve &curve1, const Curve &curve2, bool forceOversize) {	
	auto dist = fc::distance_cuda(curve1, curve2, eps, forceOversize);

	return dist;
}
