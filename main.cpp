//Jose Joseph Thandapral
//CS5330
//Project 3 :  Real-time 2-D Object Recognition
//
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int getstring(FILE* fp, char os[]);
int getint(FILE* fp, int* v);
int getfloat(FILE* fp, float* v);
int append_image_data_csv(char* filename, char* image_filename, std::vector<float>& image_data, int reset_file);
int read_image_data_csv(char* filename, std::vector<char*>& filenames, std::vector<std::vector<float>>& data, int echo_file);

//Scaled_euclidean distance calculator
float scaled_euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2, const std::vector<float>& stdevs) {
    float sum = 0.0;
    for (int i = 0; i < v1.size(); i++) {
        float scaled_diff = (v1[i] - v2[i]) / stdevs[i];
        sum += scaled_diff * scaled_diff;
    }
    return std::sqrt(sum);
}

//2pass algorithm function definition
void labelImage(Mat& binaryImage, Mat& labeledImage)
{
    // Define label matrix
    Mat labelImage = Mat::zeros(binaryImage.size(), CV_32SC1);

    // Initialize label counter
    int labelCount = 1;

    // Define neighborhood offset
    int offset[] = { -1, 0, 1 };

    // First pass
    for (int i = 0; i < binaryImage.rows; i++)
    {
        for (int j = 0; j < binaryImage.cols; j++)
        {
            if (binaryImage.at<uchar>(i, j) == 255)
            {
                vector<int> neighbors;
                for (int m = 0; m < 3; m++)
                {
                    for (int n = 0; n < 3; n++)
                    {
                        int row = i + offset[m];
                        int col = j + offset[n];
                        if (row >= 0 && col >= 0 && row < binaryImage.rows && col < binaryImage.cols)
                        {
                            int neighborLabel = labelImage.at<int>(row, col);
                            if (neighborLabel > 0)
                            {
                                neighbors.push_back(neighborLabel);
                            }
                        }
                    }
                }
                if (neighbors.empty())
                {
                    labelImage.at<int>(i, j) = labelCount++;
                }
                else
                {
                    int minNeighbor = *min_element(neighbors.begin(), neighbors.end());
                    labelImage.at<int>(i, j) = minNeighbor;
                    for (auto neighbor : neighbors)
                    {
                        if (neighbor != minNeighbor)
                        {
                            for (int m = 0; m < binaryImage.rows; m++)
                            {
                                for (int n = 0; n < binaryImage.cols; n++)
                                {
                                    if (labelImage.at<int>(m, n) == neighbor)
                                    {
                                        labelImage.at<int>(m, n) = minNeighbor;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Second pass
    vector<Vec3b> colors(labelCount);
    for (int i = 0; i < labelCount; i++)
    {
        colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }
    labeledImage = cv::Mat::zeros(binaryImage.size(), CV_8UC3);
    for (int i = 0; i < binaryImage.rows; i++)
    {
        for (int j = 0; j < binaryImage.cols; j++)
        {
            int label = labelImage.at<int>(i, j);
            labeledImage.at<Vec3b>(i, j) = colors[label];
        }
    }
}

int main(int argc, char* argv[])
{
    //create file name and image label name
    char filename[] = "object_data.csv";
    char image_filename[] = "object_1";
    char input = 1;
    vector<char*>objnames;
    vector<std::vector<float>>data;

    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();

    // Replace the IP address and port number with that of your phone's camera
    std::string ip_address = "127.0.0.1";
    std::string port_number = "6666";
    Mat segImg;
    Mat cleanImg;
    Mat frame, fgMask;
    double huMoments[7];
    // Create a VideoCapture object to connect to the camera stream
    VideoCapture capture("http://" + ip_address + ":" + port_number + "/video");
    if (!capture.isOpened()) {

        std::cout << "Error opening camera stream" << std::endl;
        return -1;
    }
    cout << "1. Training Mode 2. Classifier Mode. Enter(1/2):";
    cin >> input;
    while (true)
    {
        while (input == '1')
        {
            cout << "\nEnter the object name:";
            cin >> image_filename;
            while (true) {
                capture >> frame;
                if (frame.empty())
                    break;

                //Task 1:Threshold the input video
                //do not update the background model
                pBackSub->apply(frame, fgMask, 0);
                //get the frame number and write it on the current frame
                rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
                    cv::Scalar(255, 255, 255), -1);
                stringstream ss;
                ss << capture.get(CAP_PROP_POS_FRAMES);
                string frameNumberString = ss.str();
                putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                //show the current frame and the fg masks
                imshow("Frame", frame);
                imshow("FG Mask", fgMask);

                //get the input from the keyboard
                int keyboard = waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                    break;
            }
            while (true)
            {
                vector<float> object_features(0);
                capture >> frame;
                if (frame.empty())
                    break;

                //Task 1:Threshold the input video
                //do not update the background model
                pBackSub->apply(frame, fgMask, 0);
                //get the frame number and write it on the current frame
                rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
                    cv::Scalar(255, 255, 255), -1);
                stringstream ss;
                ss << capture.get(CAP_PROP_POS_FRAMES);
                string frameNumberString = ss.str();
                putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                //show the current frame and the fg masks
                imshow("Frame", frame);
                imshow("FG Mask", fgMask);

                //Task2:Clean up the binary image
                // apply morphological operations to clean up noise in the background and fill up the foreground image
                Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5)); // define a 5x5 elliptical structuring element

                morphologyEx(fgMask, cleanImg, MORPH_OPEN, kernel, Point(-1, -1), 1); // opening operation to remove small background noise
                morphologyEx(cleanImg, cleanImg, MORPH_CLOSE, kernel, Point(-1, -1), 3); // closing operation to fill small holes in foreground and clean its boundary
                morphologyEx(cleanImg, cleanImg, MORPH_ERODE, kernel, Point(-1, -1), 1); // erosion to further remove noise in the background
                //Show the cleaned up image
                imshow("Cleaned Up Image", cleanImg);

                //Task 3:Segmenting the regions using 2-pass algorithm
                //Mat segImg;
                labelImage(cleanImg, segImg);
                //Show the segmented image-Uncomment below line to see segmentation region map
                //imshow("Segmented Image", segImg);

                // Find contours in the labeled image
                RNG rng(12345);
                vector<vector<Point>> contours;
                findContours(cleanImg, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
                // Find the contour with the largest area
                double maxArea = -1;
                int maxIdx = -1;
                for (int i = 0; i < contours.size(); i++)
                {
                    double area = contourArea(contours[i]);
                    if (area > maxArea)
                    {
                        maxArea = area;
                        maxIdx = i;
                    }
                }
                // Draw bounding boxes around the contours on the original image
                Mat outputImage = frame.clone();
                if (maxIdx >= 0)
                {
                    Rect bbox = boundingRect(contours[maxIdx]);
                    rectangle(outputImage, bbox, Scalar(0, 255, 0), 2);
                }

                // Show the output image
                //imshow("Bounding Boxes", outputImage);

                // Calculate the moments of the largest contour
                Moments moments = cv::moments(contours[maxIdx]);

                // Calculate the centroid of the largest contour
                Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

                // Calculate the central moments of the largest contour
                double mu11 = moments.mu11 / moments.m00;
                double mu20 = moments.mu20 / moments.m00;
                double mu02 = moments.mu02 / moments.m00;

                // Calculate the orientation angle of the largest contour
                double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);

                // Calculate the ALCM using the orientation angle
                double cosTheta = cos(theta);
                double sinTheta = sin(theta);
                double x1 = centroid.x + cosTheta * mu20 + sinTheta * mu11;
                double y1 = centroid.y + cosTheta * mu11 + sinTheta * mu02;
                double x2 = centroid.x - cosTheta * mu20 - sinTheta * mu11;
                double y2 = centroid.y - cosTheta * mu11 - sinTheta * mu02;

                // Draw the ALCM on the original image
                //outputImage = frame.clone();
                line(outputImage, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

                // Show the output image
                imshow("ALCM", outputImage);

                // Calculate Hu moments
                cv::Moments mu = cv::moments(contours[maxIdx]);
                cv::HuMoments(mu, huMoments);

                // Print Hu moments
                std::cout << "\nHu Moments: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << huMoments[i] << " ";
                }

                double mu20_norm = moments.m20 / pow(moments.m00, 2.0 / 2.0);
                double mu02_norm = moments.m02 / pow(moments.m00, 2.0 / 2.0);
                double mu11_norm = moments.m11 / pow(moments.m00, 2.0 / 2.0);

                // Print normalized central moments
                std::cout << "\nmu20_norm mu02_norm mu11_norm: ";
                std::cout << mu20_norm << " " << mu02_norm << " " << mu11_norm;

                //get the input from the keyboard
                int keyboard = waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                {
                    for (int i = 0; i < 7; i++) {

                        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
                        object_features.push_back(huMoments[i]);
                    }
                    std::cout << "\n\n\nHu Moments: ";
                    for (int i = 0; i < 7; i++) {
                        std::cout << huMoments[i] << " ";
                    }
                    mu20_norm = copysign(1.0, mu20_norm) * log10(abs(mu20_norm));
                    mu02_norm = copysign(1.0, mu02_norm) * log10(abs(mu02_norm));
                    mu11_norm = copysign(1.0, mu11_norm) * log10(abs(mu11_norm));
                    std::cout << "\n\nmu20_norm mu02_norm mu11_norm: ";
                    std::cout << mu20_norm << " " << mu02_norm << " " << mu11_norm;
                    object_features.push_back(mu20_norm);
                    object_features.push_back(mu02_norm);
                    object_features.push_back(mu11_norm);
                    append_image_data_csv(filename, image_filename, object_features, 0);
                    break;
                }
            }
            cout << "\n1. Training Mode 2. Classifier Mode 3.Exit Enter(1/2/3):";
            cin >> input;
        }
        while (input == '2')
        {
            cv::destroyAllWindows();
            vector<vector<float>> featureV;
            vector<vector<string>> labelV;

            vector<float> object1_features(0);

            vector<char*> filenames;
            //cout << "\n";
            read_image_data_csv(filename, filenames, featureV, 0);
            //cout << "\n";
            for (int i = 0; i < filenames.size(); ++i)
            {
                cout << "\n" << filenames[i] << " ";
                for (int j = 0; j < featureV[i].size(); ++j)
                    cout << " " << featureV[i][j] << " ";
            }
            while (true) {
                capture >> frame;
                if (frame.empty())
                    break;

                //Task 1:Threshold the input video
                //do not update the background model
                pBackSub->apply(frame, fgMask, 0);
                //get the frame number and write it on the current frame
                rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
                    cv::Scalar(255, 255, 255), -1);
                stringstream ss;
                ss << capture.get(CAP_PROP_POS_FRAMES);
                string frameNumberString = ss.str();
                putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                //show the current frame and the fg masks
                imshow("Frame", frame);
                imshow("FG Mask", fgMask);

                //get the input from the keyboard
                int keyboard = waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                    break;
            }
            while (true)
            {
                Rect bbox;
                string topLabel;
                //Task2:Clean up the binary image
                // apply morphological operations to clean up noise in the background and fill up the foreground image
                Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5)); // define a 5x5 elliptical structuring element

                morphologyEx(fgMask, cleanImg, MORPH_OPEN, kernel, Point(-1, -1), 1); // opening operation to remove small background noise
                morphologyEx(cleanImg, cleanImg, MORPH_CLOSE, kernel, Point(-1, -1), 3); // closing operation to fill small holes in foreground and clean its boundary
                morphologyEx(cleanImg, cleanImg, MORPH_ERODE, kernel, Point(-1, -1), 1); // erosion to further remove noise in the background
                //Show the cleaned up image
                imshow("Cleaned Up Image", cleanImg);

                //Task 3:Segmenting the regions using 2-pass algorithm
                //Mat segImg;
                labelImage(cleanImg, segImg);
                //Show the segmented image
                //imshow("Segmented Image", segImg);

                // Find contours in the labeled image
                RNG rng(12345);
                vector<vector<Point>> contours;
                findContours(cleanImg, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
                // Find the contour with the largest area
                double maxArea = -1;
                int maxIdx = -1;
                for (int i = 0; i < contours.size(); i++)
                {
                    double area = contourArea(contours[i]);
                    if (area > maxArea)
                    {
                        maxArea = area;
                        maxIdx = i;
                    }
                }
                // Draw bounding boxes around the contours on the original image
                Mat outputImage = frame.clone();
                if (maxIdx >= 0)
                {
                    bbox = boundingRect(contours[maxIdx]);
                    rectangle(outputImage, bbox, Scalar(0, 255, 0), 2);
                }

                // Show the output image
                //imshow("Bounding Boxes", outputImage);

                // Calculate the moments of the largest contour
                Moments moments = cv::moments(contours[maxIdx]);

                // Calculate the centroid of the largest contour
                Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

                // Calculate the central moments of the largest contour
                double mu11 = moments.mu11 / moments.m00;
                double mu20 = moments.mu20 / moments.m00;
                double mu02 = moments.mu02 / moments.m00;

                // Calculate the orientation angle of the largest contour
                double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);

                // Calculate the ALCM using the orientation angle
                double cosTheta = cos(theta);
                double sinTheta = sin(theta);
                double x1 = centroid.x + cosTheta * mu20 + sinTheta * mu11;
                double y1 = centroid.y + cosTheta * mu11 + sinTheta * mu02;
                double x2 = centroid.x - cosTheta * mu20 - sinTheta * mu11;
                double y2 = centroid.y - cosTheta * mu11 - sinTheta * mu02;

                // Draw the ALCM on the original image
                //outputImage = frame.clone();
                line(outputImage, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

                // Show the output image
                imshow("ALCM", outputImage);

                // Calculate Hu moments
                cv::Moments mu = cv::moments(contours[maxIdx]);
                cv::HuMoments(mu, huMoments);

                // Print Hu moments
                /*std::cout << "\nHu Moments: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << huMoments[i] << " ";
                }*/

                double mu20_norm = moments.m20 / pow(moments.m00, 2.0 / 2.0);
                double mu02_norm = moments.m02 / pow(moments.m00, 2.0 / 2.0);
                double mu11_norm = moments.m11 / pow(moments.m00, 2.0 / 2.0);

                // Print normalized central moments
                //std::cout << "\nmu20_norm mu02_norm mu11_norm: ";
                //std::cout << mu20_norm << " " << mu02_norm << " " << mu11_norm;

                //get the input from the keyboard
                int keyboard = waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                {
                    for (int i = 0; i < 7; i++) {

                        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
                        object1_features.push_back(huMoments[i]);
                    }
                    /*std::cout << "\n\n\nHu Moments: ";
                    for (int i = 0; i < 7; i++) {
                        std::cout << huMoments[i] << " ";
                    }*/
                    mu20_norm = copysign(1.0, mu20_norm) * log10(abs(mu20_norm));
                    mu02_norm = copysign(1.0, mu02_norm) * log10(abs(mu02_norm));
                    mu11_norm = copysign(1.0, mu11_norm) * log10(abs(mu11_norm));
                    object1_features.push_back(mu20_norm);
                    object1_features.push_back(mu02_norm);
                    object1_features.push_back(mu11_norm);
                    //std::cout << "\n\nmu20_norm mu02_norm mu11_norm: ";
                    //std::cout << mu20_norm << " " << mu02_norm << " " << mu11_norm;

                    vector<pair<float, string>> distances;

                    // calculate the standard deviation for each feature dimension
                    std::vector<float> stdevs(featureV[0].size(), 0);
                    for (int i = 0; i < featureV[0].size(); i++) {
                        float mean = 0.0;
                        for (const auto& fv : featureV) {
                            mean += fv[i];
                        }
                        mean /= featureV.size();

                        float var = 0.0;
                        for (const auto& fv : featureV) {
                            var += (fv[i] - mean) * (fv[i] - mean);
                        }
                        var /= featureV.size();

                        stdevs[i] = std::sqrt(var);
                    }

                    for (int i = 0; i < filenames.size(); ++i)
                    {
                        float sed = scaled_euclidean_distance(object1_features, featureV[i], stdevs);
                        distances.emplace_back(sed, filenames[i]);
                    }
                    sort(distances.begin(), distances.end());
                    cout << "\nTop match:" << distances[0].second;
                    cout << "\n\nTop match based on knn:" << distances[0].second;
                    break;

                    topLabel = distances[0].second;

                    //Output of knn based classfier
                    // Prepare query and train descriptors
                    std::vector<cv::Mat> queryDescriptors, trainDescriptors;
                    for (int i = 0; i < featureV.size(); ++i) {
                        cv::Mat featureMat = cv::Mat(featureV[i]).reshape(1, 1);
                        trainDescriptors.push_back(featureMat);
                    }
                    cv::Mat featureMat1 = cv::Mat(object1_features).reshape(1, 1);
                    queryDescriptors.push_back(featureMat1);
                    // Run KNN matching
                    int k = 2;
                    std::vector<std::vector<cv::DMatch>> matches;
                    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
                    matcher->knnMatch(queryDescriptors, trainDescriptors, matches, k);

                    //print the top match based on knn nearest neighbour
                    //cout << "\n\nTop match based on knn:" << labelV(matches[0].begin());

                    // Get the position for the label
                    cv::Point textOrg(bbox.x, bbox.y - 10);
                    line(outputImage, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
                    // Draw the label around the bounding box
                    cv::putText(outputImage, topLabel, textOrg, FONT_HERSHEY_COMPLEX, outputImage.cols / 500, Scalar({ 250,200,0 }), outputImage.cols / 30);

                    // Show the output image
                    imshow("Top Match", outputImage);
                }
                
            }

            cout << "\n1. Training Mode 2. Classifier Mode 3.Exit Enter(1/2/3):";
            cin >> input;
        }
        
    }
    if (input == '3')
    {
        //get the input from the keyboard
        while (cv::waitKey(1) != 'q');
        cv::destroyAllWindows();
        return 0;
    }
}
