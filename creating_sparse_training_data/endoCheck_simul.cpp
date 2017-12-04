//============================================================================
// Name        : EndoCheck.cpp
// Author      : Sebastian Bodenstedt
// Version     :
// Copyright   : 
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <Interfaces/MainWindowInterface.h>
#include <Interfaces/MainWindowEventInterface.h>
#include <Interfaces/ApplicationHandlerInterface.h>
#include <gui/GUIFactory.h>
#include <HRM.h>
#include <ImageProcessing.h>
#include <Image/ImageProcessor.h>
#include <Image/ByteImage.h>
#include <Image/PrimitivesDrawer.h>
#include <VideoCapture/BitmapSequenceCapture.h>
#include "endoCheck.h"
#include <qframe.h>
#include <qfiledialog.h>
#include <Image/ByteImage.h>
#include <Image/IplImageAdaptor.h>
#include <iostream>

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
#include <numeric>
#include <fstream>
#include <ctime>
using namespace cv;

//#define SeqName "/org/share/MediData/MedData/Tier/Leber/20140802/Borsti/Overview"
#define SourceFolder "/media/INTENSO/Schwein-2-8/Borsti/SchwenkDefLong/"


EndoCheck::EndoCheck(bool computeHRM, char* path) :
        CQTWindow(100, 100), rng(time(0))
{
	m_computeHRM = computeHRM;
	m_path = path;
}

EndoCheck::~EndoCheck()
{

}
void EndoCheck::nextClick()
{
	next = true;
}


void EndoCheck::computeSurf(Mat img_object, Mat img_scene)
{
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector(minHessian);
    //SiftFeatureDetector detector;


    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector.detect(img_object, keypoints_object, maskLeft);
    detector.detect(img_scene, keypoints_scene, maskRight);

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
    //SiftDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor.compute(img_object, keypoints_object, descriptors_object);
    extractor.compute(img_scene, keypoints_scene, descriptors_scene);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors_object, descriptors_scene, matches);

    double max_dist = 0;
    double min_dist = 150;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors_object.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    //printf("-- Max dist : %f \n", max_dist);
    //printf("-- Min dist : %f \n", min_dist);

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector<DMatch> good_matches;

    for (int i = 0; i < descriptors_object.rows; i++)
    {
        float x1 = keypoints_object.at(matches[i].queryIdx).pt.x;
        float y1 = keypoints_object.at(matches[i].queryIdx).pt.y;
        float x2 = keypoints_scene.at(matches[i].trainIdx).pt.x;
        float y2 = keypoints_scene.at(matches[i].trainIdx).pt.y;

        if ( matches[i].distance < 5 * min_dist && abs(y1 - y2) < 3  && x1 - x2 < 50 && x1 - x2 > 0)
        {
            uchar max = 0;
            for(int dx=-2; dx <=2;dx++)
                for(int dy=-2; dy <=2;dy++)
                {
                    uchar value = img_object.at<uchar>(Point(x1+dx,y1+dy));

                    if (value > max)
                        max = value;
                }
            if(max > 230)
                continue;

            good_matches.push_back(matches[i]);

            disparities[(int)x1+(int)y1*width] = x1 - x2;
            disparities[1+(int)x1+(int)y1*width] = x1 - x2;
            disparities[(int)x1+(1+(int)y1)*width] = x1 - x2;
            disparities[1+(int)x1+(1+(int)y1)*width] = x1 - x2;

            //disparities.push_back(keypoints_object.at(matches[i].queryIdx).pt.x - keypoints_scene.at(matches[i].trainIdx).pt.x);
        }
    }

    float sumDif = 0;
    int temp;


    vector<double> v;


    /*for(int i = 0; i < good_matches.size(); i++)
    {
        temp = keypoints_object.at(good_matches.at(i).queryIdx).pt.y - keypoints_scene.at(good_matches.at(i).trainIdx).pt.y;
        if(temp < 10 && temp > -10)
        {
            v.push_back(keypoints_object.at(good_matches.at(i).queryIdx).pt.y - keypoints_scene.at(good_matches.at(i).trainIdx).pt.y);
            //cout << v.at(v.size()-1) << endl;
        }
    }

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size() - mean * mean);*/

    //cout << "Unterschied in der Y-Achse ist im Schnitt = " << mean << "Pixel" << endl;
    //cout << "Standardabweichung ist = " << stdev << "Pixel" << endl;

    cout << "Anzahl guter Matches Surf = " << good_matches.size()  << endl;

      Mat img_matches;
      drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                   good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //imshow( "Good Matches & Object detection", img_matches );

    imwrite("left.bmp",img_scene);

    if(good_matches.size() == 0)
        has_matches = false;

    //cv::waitKey(0);
}


void EndoCheck::computeOrb(Mat img_object, Mat img_scene)
{
    SiftFeatureDetector detector; //OrbFeatureDetector detector;SurfFeatureDetector
    vector<KeyPoint> keypoints1;
    detector.detect(img_object, keypoints1, maskLeft);
    vector<KeyPoint> keypoints2;
    detector.detect(img_scene, keypoints2, maskRight);

    SiftDescriptorExtractor extractor; //OrbDescriptorExtractor extractor; SurfDescriptorExtractor extractor;
    Mat descriptors_1, descriptors_2;
    extractor.compute( img_object, keypoints1, descriptors_1 );
    extractor.compute( img_scene, keypoints2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors with a brute force matcher
    //BFMatcher matcher(NORM_L2, true);   //BFMatcher matcher(NORM_L2);
    //FlannBasedMatcher matcher(new cv::flann::LshIndexParams(5, 24, 2));
    FlannBasedMatcher matcher;
    vector< DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);


    double max_dist = 0;
    double min_dist = 150;
    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }


    vector< DMatch > good_matches;

    for(int i=0; i<int(matches.size()); i++){

        float x1 = keypoints1.at(matches[i].queryIdx).pt.x;
        float y1 = keypoints1.at(matches[i].queryIdx).pt.y;
        float x2 = keypoints2.at(matches[i].trainIdx).pt.x;
        float y2 = keypoints2.at(matches[i].trainIdx).pt.y;

        if ( matches[i].distance < 5 * min_dist && abs(y1 - y2) < 3  && x1 - x2 < 50 && x1 - x2 > 0)
        {
            uchar max = 0;
            for(int dx=-2; dx <=2;dx++)
                for(int dy=-2; dy <=2;dy++)
                {
                    uchar value = img_object.at<uchar>(Point(x1+dx,y1+dy));

                    if (value > max)
                        max = value;
                }
            if(max > 230)
                continue;


            good_matches.push_back(matches[i]);
            disparities[(int)x1+(int)y1*width] = x1 - x2;
            disparities[1+(int)x1+(int)y1*width] = x1 - x2;
            disparities[(int)x1+(1+(int)y1)*width] = x1 - x2;
            disparities[1+(int)x1+(1+(int)y1)*width] = x1 - x2;

        }
    }
    cout << "Anzahl guter Matches SIFT= " << good_matches.size() <<  endl;
      Mat img_matches;
      drawMatches( img_object, keypoints1, img_scene, keypoints2,
                   good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imshow( "Good Matches & Object detection", img_matches );

    //select one match for specularity simulation
    if(good_matches.size() > 0)
    {
        DMatch specular_match = good_matches.at(rng.uniform(0,good_matches.size()));
        specular_match_position[0] = keypoints1.at(specular_match.queryIdx).pt.x;
        specular_match_position[1] = keypoints1.at(specular_match.queryIdx).pt.y;

        specular_disparity = keypoints1.at(specular_match.queryIdx).pt.x - keypoints2.at(specular_match.trainIdx).pt.x;

    }
    else
        has_matches = false;

    //cv::waitKey(0);

}



bool EndoCheck::Run()
{
	char left[100];
	char right[100];

    sprintf(left, "%sleft_normal_0.bmp", m_path);
    sprintf(right, "%sright_normal_0.bmp", m_path);


	CBitmapSequenceCapture capture(left, right);
	if (!capture.OpenCamera())
		printf("Couldn't open camera\n");
    width = capture.GetWidth();
    height = capture.GetHeight();
	QPushButton* bNext = new QPushButton(this);
	bNext->setText("Next");
	bNext->setFixedHeight(30);
	bNext->setFixedWidth(70);
	bNext->move(0, 1080);
	connect(bNext, SIGNAL(clicked()), this, SLOT(nextClick()));
    Show();
    //CImageProcessing* img_proc = new CImageProcessing("/org/share/MediData/MedData/Tier/Darmvermessung/Real/cameras2.txt");
    //CImageProcessing* img_proc = new CImageProcessing("/org/share/MediData/MedData/Tier/Leber/26-02-14/Calib/camera.txt");
	const CByteImage::ImageType type = capture.GetType();

    next = true;

	CByteImage* inImages[2];
	inImages[0] = new CByteImage(width, height, type);
	inImages[1] = new CByteImage(width, height, type);
	CByteImage* img_out = new CByteImage(width, height, type);
	CByteImage* img_out2 = new CByteImage(width, height, type);

    disparities = new double[width*height];

    for (int i=0; i<width*height; i++) {
        disparities[i] = 0;    // Initialize all elements to zero.
    }

	//lese FeatureMasken ein
    maskLeft = imread( "/org/share/MediData/MedData/Tier/Darmvermessung/Real/maskleft_rect.bmp", CV_LOAD_IMAGE_GRAYSCALE );
    maskRight = imread( "/org/share/MediData/MedData/Tier/Darmvermessung/Real/maskright_rect.bmp", CV_LOAD_IMAGE_GRAYSCALE );

    Mat specular_highlight = cv::imread("/org/share/MediData/MedData/Simluation/dispnet/specular_highlight.png", cv::IMREAD_UNCHANGED);

	//setup für die Speicherung der Bilder
	int frameNumber = 1780;

	while (this->isVisible())
	{
		QApplication::processEvents();

		if (next)
		{
			if (!capture.CaptureImage(inImages))
				break;
            next = true;

            has_matches = true;

            IplImage* cvImages[2] = { IplImageAdaptor::Adapt(inImages[0]), IplImageAdaptor::Adapt(inImages[1]) };

            Mat img_object(cvImages[0]);
            Mat img_scene(cvImages[1]);

            if (!img_object.data || !img_scene.data)
            {
                std::cout << " --(!) Error reading images " << std::endl;
                return -1;
            }

            cout << "image no.: " << frameNumber << endl;

            Mat left(img_object.rows/2,img_object.cols,img_object.type());
            Mat right(img_object.rows/2,img_object.cols,img_object.type());

            for(int y=0; y < img_object.rows/2; y++)
               for(int x=0; x < img_object.cols; x++)
               {
                   left.at<Vec3b>(Point(x,y)) = img_object.at<Vec3b>(Point(x,y*2));
                   right.at<Vec3b>(Point(x,y)) = img_scene.at<Vec3b>(Point(x,y*2));
               }

            cv::resize(left, img_object, Size(img_object.cols, img_object.rows));
            cv::resize(right, img_scene, Size(img_object.cols, img_object.rows));


            //img_proc->PreProcessing(inImages[0], inImages[1]);

			CvMat* grayImages[2] = { cvCreateMat(height, width, CV_8UC1), cvCreateMat(height, width, CV_8UC1) };

            CvMat* colorImages[2] = { cvCreateMat(height, width, CV_8UC3), cvCreateMat(height, width, CV_8UC3) };

            cvCvtColor(cvImages[0], colorImages[0], CV_BGR2RGB);
            cvCvtColor(cvImages[1], colorImages[1], CV_BGR2RGB);

            cvCvtColor(colorImages[0], grayImages[0], CV_BGR2GRAY);
            cvCvtColor(colorImages[1], grayImages[1], CV_BGR2GRAY);

            Mat grey_left(grayImages[0]);
            Mat grey_right(grayImages[1]);

            computeSurf(grey_left, grey_right);
            computeOrb(grey_left, grey_right);

            Mat out_left(colorImages[0]);
            Mat out_right(colorImages[1]);


            Mat transformed_highlight;
            cv::resize(specular_highlight, transformed_highlight, cv::Size(rng.uniform(20,100),rng.uniform(20,100)));
            //cv::imshow("transformed_highlight", transformed_highlight);

            Vec3i highlight_position(specular_match_position[0]-transformed_highlight.cols/2+rng.uniform(-20,20),
                   specular_match_position[1]-transformed_highlight.rows/2+rng.uniform(-20,20));



            Mat highlights_left(out_left.size(), CV_8UC4);
            Mat highlights_right(out_left.size(), CV_8UC4);
            highlights_left.setTo(0);
            highlights_right.setTo(0);


            //blit hightlight
            if(transformed_highlight.cols + highlight_position[0] < highlights_left.cols &&
             transformed_highlight.rows + highlight_position[1] + specular_disparity < highlights_left.rows &&
             highlight_position[0] > 0 && highlight_position[1] > 0)
                transformed_highlight.copyTo(highlights_left(Rect(highlight_position[0], highlight_position[1], transformed_highlight.cols, transformed_highlight.rows)));
            else
                transformed_highlight.copyTo(highlights_left(Rect(200, 200, transformed_highlight.cols, transformed_highlight.rows)));

            Mat rot = cv::getRotationMatrix2D(Point(highlight_position[0]+transformed_highlight.cols/2,
                highlight_position[1]+transformed_highlight.rows/2), rng.uniform(0,360),1);
            cv::warpAffine(highlights_left,highlights_left,rot,highlights_left.size());

            int light_disp = specular_disparity * rng.uniform(0.3f,0.8f);
            int t[] = {};
            Mat translate_right = (Mat_<float>(2,3) << 1,0,-light_disp,0,1,0);
            cv::warpAffine(highlights_left,highlights_right,translate_right,highlights_left.size());

            Mat left_hsv;
            Mat right_hsv;
            cv::cvtColor(out_left, left_hsv, CV_RGB2HSV);
            cv::cvtColor(out_right, right_hsv, CV_RGB2HSV);

            //alpha blending
            for(int y=0; y < out_left.rows; y++)
               for(int x=0; x < out_left.cols; x++)
               {
                   const Vec4b &highlight_left = highlights_left.at<Vec4b>(Point(x,y));
                   Vec3b &color_left = out_left.at<Vec3b>(Point(x,y));
                   Vec3b &hsv_left = left_hsv.at<Vec3b>(Point(x,y));
                   const Vec4b &highlight_right = highlights_right.at<Vec4b>(Point(x,y));
                   Vec3b &color_right = out_right.at<Vec3b>(Point(x,y));
                   Vec3b &hsv_right = right_hsv.at<Vec3b>(Point(x,y));

                   float alpha_left = highlight_left[3]/255.0f;
                   alpha_left = alpha_left*alpha_left;

                   float alpha_right = highlight_right[3]/255.0f;
                   alpha_right = alpha_right*alpha_right;

                   hsv_left[2] = hsv_left[2]*(1-alpha_left) + alpha_left*255;
                   hsv_left[1] = hsv_left[1]*(1-alpha_left);


                   hsv_right[2] = hsv_right[2]*(1-alpha_right) + alpha_right*255;
                   hsv_right[1] = hsv_right[1]*(1-alpha_right);

               }

            cv::cvtColor(left_hsv,out_left, CV_HSV2RGB);
            cv::cvtColor(right_hsv,out_right, CV_HSV2RGB);

            cv::imshow("out_left", out_left);
            cv::imshow("out_right", out_right);

            if(has_matches)
            {
                char file[200];
                sprintf(file, "/local_home/daniel/sparse_pig/x/left_depth_%i.disp", frameNumber);

                std::ofstream OutFile;
                OutFile.open(file, ios::out | ios::binary);

                for(int i = 0; i < width*height; i++)
                {
                    OutFile.write( (char*)&disparities[i], sizeof(double));
                }
                for (int i=0; i<width*height; i++) {
                    disparities[i] = 0;    // Initialize all elements to zero.
                }

                sprintf(file, "/local_home/daniel/sparse_pig/x/left_normal_%i.png", frameNumber);
                //inImages[0]->SaveToFile(file);
                imwrite(file, out_left);
                sprintf(file, "/local_home/daniel/sparse_pig/x/right_normal_%i.png", frameNumber);
                //inImages[0]->SaveToFile(file);
                imwrite(file, out_right);

                OutFile.close();
            }

            cv::waitKey(0);



            frameNumber++;

        }
    }

//	delete img_proc;
	delete inImages[0];
	delete inImages[1];
	delete img_out;
	delete img_out2;

	return true;
}

int main(int argc, char** argv)
{
	CQTApplicationHandler* qt_application_handler = new CQTApplicationHandler(argc, argv);

	qt_application_handler->Reset();
	EndoCheck* window;

	if(argc == 2)
		window = new EndoCheck(true, argv[1]);
	else if(argc == 3 && atoi(argv[2]) == 1)
		window = new EndoCheck(true, argv[0]);
	else if(argc == 3 && atoi(argv[2]) == 0)
		window = new EndoCheck(false, argv[1]);
	else
		return EXIT_FAILURE;

	window->Run();

	return EXIT_SUCCESS;
}

