#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <sstream>


struct Point {
	std::vector<double> coordinates;
	
	inline Point() {}
	
	inline Point(const std::vector<double> &in) : coordinates{in} {}
	
	inline Point& operator-=(const Point &point)
	{
		for(unsigned long i = 0; i < coordinates.size(); ++i){
			coordinates[i] -= point.getCoordinates()[i];
		}
		return *this;
	}

	inline Point operator-(const Point &point) const
	{
		auto result = *this;
		for(unsigned long i = 0; i < coordinates.size(); ++i){
			result.coordinates[i] -= point.getCoordinates()[i];
		}

		return result;
	}

	inline Point& operator+=(const Point &point)
	{
		for(unsigned long i = 0; i < coordinates.size(); ++i){
			coordinates[i] += point.getCoordinates()[i];
		}
		return *this;
	}

	inline Point operator+(const Point &point) const
	{
		auto result = *this;
		for(unsigned long i = 0; i < coordinates.size(); ++i){
			result.coordinates[i] += point.getCoordinates()[i];
		}

		return result;
	}

	inline Point operator*(const double mult) const
	{
		Point res;
		res.coordinates = std::vector<double> (coordinates.size());
		for(unsigned long i = 0; i < coordinates.size(); ++i){
			res.getCoordinates()[i] = coordinates[i] * mult;
		}
		return res;
	}

	inline Point& operator/=(double distance)
	{
		for(unsigned long i = 0; i < coordinates.size(); ++i){
			coordinates[i] /= distance;
		}
		return *this;
	}

	inline double dist_sqr(const Point &point) const
	{
		double result = 0;
		double distsigned;
		for(unsigned long i = 0; i < coordinates.size(); ++i){
			distsigned = coordinates[i] - point.getCoordinates()[i];
			result += distsigned * distsigned;
		}
		return result;
	}

	inline double dist(const Point &point) const
	{
		return std::sqrt(dist_sqr(point));
	}
	
	inline std::vector<double>& getCoordinates() {
		return this->coordinates;
	}
	inline const std::vector<double>& getCoordinates() const {
		return this->coordinates;
	}
	inline double operator[](std::size_t i) const { return coordinates[i]; }
	inline int size() const{return coordinates.size();}
private:
	
};
typedef std::vector<Point> Points;

typedef std::vector<Points> multiPoints;

std::ostream& operator<<(std::ostream&, const Point&);

struct Interval
{
	double begin;
	double end;

	Interval()
		: begin(1.),
		  end(0.) {}

	Interval(double begin, double end)
		: begin(begin),
		  end(end) {}

	inline bool operator<(const Interval &other) const {
		return begin < other.begin || (begin == other.begin && end < other.end);
	}

	inline bool is_empty() const { return begin > end; }
	inline bool intersects(const Interval &other) const
	{
		if (is_empty() or other.is_empty()) { return false; }

		return (other.begin >= begin and other.begin <= end) or
			(other.end >= begin and other.end <= end) or
			(other.begin <= begin and other.end >= end);
	}
};
typedef std::vector<Interval> Intervals;

std::ostream& operator<<(std::ostream&, const Interval&);

class IntersectionAlgorithm {
	template<typename T>
	inline static T pow2(T d) { return d * d; }
	inline constexpr IntersectionAlgorithm() {} // Make class static-only
	inline static bool smallDistanceAt(double interpolate, const Point &line_start, const Point &line_end, const Point &circle_center, double radius_sqr) {
		return circle_center.dist_sqr(line_start * (1. - interpolate) + line_end * interpolate) <= radius_sqr;
	}

	inline static double distanceAt(double interpolate, const Point &line_start, const Point &line_end, const Point &circle_center) {
		return circle_center.dist_sqr(line_start * (1. - interpolate) + line_end * interpolate);
	}
	static constexpr double eps = 0.001 / 4;
	static constexpr double save_eps = 0.5 * eps;
	static constexpr double save_eps_half = 0.25 * eps;
	
public:
   /*
    * Returns which section of the line segment from line_start to line_end is inside the circle given by circle_center and radius.
    * If the circle and line segment do not intersect, the result is the empty Interval (and outer is the empty Interval, too).
	* Otherwise the result is an interval [x,y] such that the distance at x and at y is at most the radius, i.e., [x,y] is a subset of the free interval. 
	* The optional output "outer" is an interval strictly containing the free interval.
	* In other words, "outer" is an interval [x',y'] containing [x,y] such that x-x', y'-y <= eps and:
	* If x = 0 then x' = -eps, while if x > 0 then the distance at x' is more than the radius.
	* If y = 1 then y' = 1+eps, while if y < 1 then the distance at y' is more than the radius.
    */
	inline static Interval intersection_interval(const Point &circle_center, double radius, const Point &line_start, const Point &line_end) {
		// The line can be represented as line_start + lambda * v
		const Point v = line_end - line_start;
		const double rad_sqr = radius * radius;
		
		// Find points p = line_start + lambda * v with
		//     dist(p, circle_center) = radius
		// <=> sqrt(p.x^2 + p.y^2) = radius
		// <=> p.x^2 + p.y^2 = radius^2
		// <=> (line_start.x + lambda * v.x)^2 + (line_start.y + lambda * v.y)^2 = radius^2
		// <=> (line_start.x^2 + 2 * line_start.x * lambda * v.x + lambda^2 * v.x^2) + (line_start.y^2 + 2 * line_start.y * lambda * v.y + lambda^2 * v.y^2) = radius^2
		// <=> lambda^2 * (v.x^2 + v.y^2) + lambda * (2 * line_start.x * v.x + 2 * line_start.y * v.y) + line_start.x^2 + line_start.y^2) - radius^2 = 0
		// let a := v.x^2 + v.y^2, 
		// let b := line_start.x * v.x + line_start.y * v.y, 
		// let c := line_start.x^2 + line_start.y^2 - radius^2
		// <=> lambda^2 * a + lambda * 2 b + c = 0
		// <=> lambda^2 + (2 b / a) * lambda + (c / a) = 0
		// <=> lambda1/2 = - (b / a) +/- sqrt((b / a)^2 - c / a)
		double dist_a = 0;
		double dist_b = 0;
		double dist_c = - pow2(radius);

		for(unsigned long i = 0; i < v.getCoordinates().size(); ++i){
			dist_a += pow2(v.getCoordinates()[i]);
			dist_b += (line_start.getCoordinates()[i] - circle_center.getCoordinates()[i]) * v.getCoordinates()[i];
			dist_c += pow2(line_start.getCoordinates()[i] - circle_center.getCoordinates()[i]);
		}

		const double a = dist_a;
		const double b = dist_b;
		const double c = dist_c;


		double mid = - b / a;
		double discriminant = pow2(mid) - c / a;

		const bool smallDistAtZero = smallDistanceAt(0., line_start, line_end, circle_center, rad_sqr);
		const bool smallDistAtOne = smallDistanceAt(1., line_start, line_end, circle_center, rad_sqr);
		bool smallDistAtMid = smallDistanceAt(mid, line_start, line_end, circle_center, rad_sqr);
		
		if (smallDistAtZero and smallDistAtOne) {
			return Interval(0, 1);
		}
		
		if (not smallDistAtMid and smallDistAtZero) {
			mid = 0.;
			smallDistAtMid = true;
		} else if (not smallDistAtMid and smallDistAtOne) {
			mid = 1.;
			smallDistAtMid = true;
		}
		
		// Here we need the guarantee that if the free interval has length at least eps
		// then at mid the distance is <=radius
		// This is an assumption about the precision of double computations
		// All remaining rules are free of such assumptions! 
		// (except for trivial ones like this: x + y and x - y have distance at most 2y up to negligible error)
		if (not smallDistAtMid) {
			return Interval(); // no intersection;
		}
		
		if (mid <= 0. and not smallDistAtZero) {
			return Interval();
		}
		if (mid >= 1. and not smallDistAtOne) {
			return Interval();
		}
		
		discriminant = std::max<double>(discriminant, 0.);
		double sqrt_discr = 0.;
		bool sqrt_discr_computed = false;
		double begin, end;
		
		if (smallDistAtZero) {
			begin = 0.;
		} else {
			sqrt_discr = std::sqrt(discriminant);
			sqrt_discr_computed = true;
			
			const double lambda1 = mid - sqrt_discr;
			const double innershift = std::min<double>(lambda1 + save_eps_half, std::min<double>(1., mid));
			const double outershift = lambda1 - save_eps_half;
			if (innershift >= outershift and smallDistanceAt(innershift, line_start, line_end, circle_center, rad_sqr) 
				and not smallDistanceAt(outershift, line_start, line_end, circle_center, rad_sqr)) {
				begin = innershift;
			}
			else {
				double left = 0., right = std::min<double>(mid, 1.);
				// invariants throughout binary search:
				//  * !smallDistanceAt(left)
				//  * smallDistanceAt(right)
				//  * 0 <= left <= right <= min(mid,1)
				// Clearly this is stays true after an iteration.
				// Why is it true in the beginning?
				// If smallDistanceAt(0.) then begin would already be set (fourth rule).
				// If !smallDistanceAt(right), then either !smallDistanceAt(mid), contradicting the very first rule, 
				//  or mid >= 1. and smallDistanceAt(1.), contradicting the third rule.
				// Finally, since !smallDistanceAt(left) we cannot have mid <= 0 by the second rule. Thus, right = min(mid,1) >= 0. = left
				double m = 0.5 * (left + right);
				while (right - left > save_eps) {
					m = 0.5 * (left + right);
					if (smallDistanceAt(m, line_start, line_end, circle_center, rad_sqr)) right = m;
					else left = m;
				}
				begin = right;
			}
		}
		
		if (smallDistAtOne) {
			end = 1.;
		} else {
			if (not sqrt_discr_computed) {
				sqrt_discr = std::sqrt(discriminant);
			}
			
			const double lambda2 = mid + sqrt_discr;
			const double innershift = std::max<double>(lambda2 - save_eps_half, std::max<double>(0., mid));
			const double outershift = lambda2 + save_eps_half;
			if (innershift <= outershift and smallDistanceAt(innershift, line_start, line_end, circle_center, rad_sqr) 
				and not smallDistanceAt(outershift, line_start, line_end, circle_center, rad_sqr)) {
				end = innershift;
			} else {
				double left = std::max<double>(mid, 0.), right = 1.;
				// invariants throughout binary search:
				//  * smallDistanceAt(left)
				//  * !smallDistanceAt(right)
				//  * max(mid,0) <= left <= right <= 1
				double m = 0.5 * (left + right);
				while (right - left > save_eps) {
					m = 0.5 * (left + right);
					if (smallDistanceAt(m, line_start, line_end, circle_center, rad_sqr)) left = m;
					else right = m;
				}
				end = left;
			}
		}
		return Interval{begin, end};
	}
    IntersectionAlgorithm(const IntersectionAlgorithm&) = delete;
    void operator=(const IntersectionAlgorithm&) = delete;
};
