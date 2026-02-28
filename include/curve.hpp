#pragma once

#include <iostream> 

#include "geometry_basics.hpp"

// Represents a trajectory. Additionally to the points given in the input file,
// we also store the length of any prefix of the trajectory.
class Curve {
public:
	typedef unsigned long index_type;

	Curve(unsigned int dimensions) : number_dimensions{dimensions} {}
    Curve(const Points& points, unsigned int dimensions);

    inline std::size_t size() const { return points.size(); }
	inline bool empty() const { return points.empty(); }
	inline std::size_t dimensions() const { return number_dimensions; }
    inline Point const& operator[](const std::size_t i) const { return points[i]; }

    inline Point front() const { return points.front(); }
    inline Point back() const { return points.back(); }

    void push_back(const Point &point);
	void recenter(const Point &point){
		printf("%f, %f\n", points[1][0], points[1][1]);
		printf("%f, %f\n", point[0], point[1]);
		for(unsigned long i = 0; i < points.size(); i++) points[i] -= point;
		printf("%f, %f\n", points[1][0], points[1][1]);
	}

	inline Points::iterator begin() { return points.begin(); }
	inline Points::iterator end() { return points.end(); }
	inline Points::const_iterator begin() const { return points.cbegin(); }
	inline Points::const_iterator end() const { return points.cend(); }
	
private:
	Points points;
	unsigned int number_dimensions;
};

class Curves : public std::vector<Curve> {
public:
	inline Curve get(std::size_t i) const {
		return this->operator[](i);
	}
};

std::ostream& operator<<(std::ostream& out, const Curve& curve);
