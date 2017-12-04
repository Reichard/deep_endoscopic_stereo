#ifndef TMI_H_
#define TMI_H_

#include <string>
#include <PreprocessingFilter/PreprocessingFilter.h>

class TMI
{
public:
    void setCalibration(std::string s);

    void setFileName(std::string s)
	{
		file_name = s;
    }
    void compute_3d_points();

private:
    std::string                  file_name;
    CStereoCalibration      m_calibration;
};


#endif
