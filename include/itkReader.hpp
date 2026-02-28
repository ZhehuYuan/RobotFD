#include "itkDirectedHausdorffDistanceImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"

#include "geometry_basics.hpp"

#include <string>
#include <vector>
#include <iostream>

Points loadITK(const std::string& path, unsigned int N_DIMS){
	using ImageType = itk::Image<unsigned char, N_DIMS>;
	using ReaderType = itk::ImageFileReader<ImageType>;
	
	
}
