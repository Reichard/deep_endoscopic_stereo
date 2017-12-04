#include "tmi.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;


void TMI::setCalibration(string s)
{
    m_calibration.LoadCameraParameters(s.c_str());
}

void TMI::compute_3d_points()
{
    const float EPSILON = 0.001f;

    //read each pixel diparity value and compute the corresponding 3d point
    //3d points are stored in x, y, 3dx, 3dy, 3dz each line
    ofstream outfile;
    outfile.open ("3dPoints.xyz");

    //read disparities
    std::ifstream infile(file_name.c_str());
    int width, height;
    infile >> width >> height;

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
        	float disp;
        	infile >> disp;

            if(disp > width || disp < EPSILON) continue;

    		Vec2d left = {x,y};
    		Vec2d right = {x-disp,y};

    		Vec3d point;
            m_calibration.Calculate3DPoint(left, right, point, true, false);

            if(isnan(point.x + point.y + point.z) || isinf(point.x + point.y + point.z)) continue;
            if(point.z < EPSILON || point.z > 1000) continue;

            outfile << x << " "<< y <<" "<< point.x <<" "<< point.y <<" "<< point.z << endl;
        }
	}
    outfile.close();
    infile.close();
}

int main(int argc, char** argv)
{
    if(argc != 3)
        cout << "Usage: " << argv[0] << " <disparity_file> <calibration_file>" << endl;
    else
    {
        TMI tmi;

        tmi.setCalibration(string(argv[2]));
        tmi.setFileName(string(argv[1]));
        tmi.compute_3d_points();
    }

}
