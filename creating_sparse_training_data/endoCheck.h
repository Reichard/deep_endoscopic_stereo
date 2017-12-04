/*
 * template.h
 *
 *  Created on: Nov 5, 2010
 *  Author: S. Bodenstedt
 *  
 * QT Template
 */

#ifndef EndoCheck_H_
#define EndoCheck_H_
#include <gui/QTApplicationHandler.h>
#include <gui/QTWindow.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qlineedit.h>
#include <QtGui>
#include "opencv2/core/core.hpp"

class EndoCheck : public CQTWindow
{
	Q_OBJECT
	
	public: 
        EndoCheck(bool computeHRM, char* path, char* out_path);
		~EndoCheck();
		bool Run();
    private:
        cv::RNG rng;
        bool has_matches;
		bool next;
		bool m_computeHRM;
        char* m_path, *m_out_path;
        cv::Mat maskLeft, maskRight;
        cv::Vec2i specular_match_position;
        float specular_disparity;
        void computeSurf(cv::Mat img_object, cv::Mat img_scene);
        void computeOrb(cv::Mat img_object, cv::Mat img_scene);
        int width, height;
        //std::vector<double> disparities;
        double* disparities;
private slots:
		void nextClick();

};
#endif
