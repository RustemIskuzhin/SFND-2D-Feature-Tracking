# SFND 2D Feature Tracking
## [Rubric](https://review.udacity.com/#!/rubrics/2549/view) Points
---
#### 1. Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end.
MidTermProject_Camera_Student.cpp
```c++
int dataBufferSize = 3;       // no. of images which are held in memory (ring buffer) at the same time
vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

// push image into data frame buffer
DataFrame frame;
frame.cameraImg = imgGray;
if  (dataBuffer.size() < dataBufferSize)
{  
    dataBuffer.push_back(frame);
    cout << "LOAD IMAGE INTO BUFFER done" << endl;
}  
else
{
    dataBuffer.erase(dataBuffer.begin());
    dataBuffer.push_back(frame);
    cout << "REPLACE IMAGE IN BUFFER done" << endl;
}
```
#### 2. Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.
MidTermProject_Camera_Student.cpp
```c++
if (detectorType.compare("SHITOMASI") == 0)
{
    detKeypointsShiTomasi(keypoints, imgGray, false);
}
else if (detectorType.compare("HARRIS") == 0)
{
    detKeypointsHarris(keypoints, imgGray, false);
}
else 
{
    detKeypointsModern(keypoints, imgGray, detectorType, false);
}
```
matching2D_Student.cpp
```c++
//FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    string windowName;
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "FAST  Detector Results";
    }  
    else if (detectorType.compare("BRISK") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "BRISK detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "BRISK  Detector Results";
    }
    else if (detectorType.compare("ORB") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "ORB detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "ORB  Detector Results";
    }  
    else if (detectorType.compare("AKAZE") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "AKAZE detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "AKAZE  Detector Results";
    }  
    else if (detectorType.compare("SIFT") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "SIFT detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
         windowName = "SIFT  Detector Results";
    }  
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
   // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
  
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
  
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
 
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
  
}
```
#### 3. Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.
MidTermProject_Camera_Student.cpp
```c++
// only keep keypoints on the preceding vehicle
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(535, 180, 180, 150);
vector<cv::KeyPoint>::iterator keypoint;
vector<cv::KeyPoint> keypoints_roi;
if (bFocusOnVehicle)
{
    for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
    {
        if (vehicleRect.contains(keypoint->pt))
        {  
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f(keypoint->pt);
            newKeyPoint.size = 1;
            keypoints_roi.push_back(newKeyPoint);
        }
    }
    keypoints =  keypoints_roi;
    cout << "IN ROI n= " << keypoints.size()<<" keypoints"<<endl;
}
```
#### 4. Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.
MidTermProject_Camera_Student.cpp
```c++
string descriptorType = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT
descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
```
matching2D_Student.cpp
```c++
// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}
```

#### 5. Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.
matching2D_Student.cpp
```c++
if (matcherType.compare("MAT_BF") == 0)
{
    int normType = cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
}
else if (matcherType.compare("MAT_FLANN") == 0)
{
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}
```
#### 6. Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.
matching2D_Student.cpp
```c++
// perform matching task
if (selectorType.compare("SEL_NN") == 0)
{ // nearest neighbor (best match)

    matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
}
else if (selectorType.compare("SEL_KNN") == 0)
{ // k nearest neighbors (k=2)
    vector<vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descSource, descRef, knn_matches, 2);
    double minDescDistRatio = 0.8;
    for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
    {

        if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
        {
            matches.push_back((*it)[0]);
        }
    }
    cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
}
```
#### 7. Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

I created a class to save the number of keypoints in a CSV file.

MidTermProject_Camera_Student.cpp
```c++
std::string FileName = "/home/workspace/SFND_2D_Feature_Matching/Output.csv";

// A class to create and write data in a csv file.
class CSVWriter
{
	std::string fileName;
	std::string delimeter;
	int linesCount;
 
public:
	CSVWriter(std::string filename, std::string delm = ",") :
			fileName(filename), delimeter(delm), linesCount(0)
	{}
	/*
	 * Member function to store a range as comma seperated value
	 */
	template<typename T>
	void addDatainRow(T first, T last);
};

/*
* This Function accepts a range and appends all the elements in the range
* to the last row, seperated by delimeter (Default is comma)
*/
template<typename T>
void CSVWriter::addDatainRow(T first, T last)
{
    std::fstream file;
    // Open the file in truncate mode if first line else in Append Mode
    file.open(fileName, std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));
    // Iterate over the range and add each lement to file seperated by delimeter.
    for (; first != last; )
    {
        file << *first;
        if (++first != last)
            file << delimeter;
        }
    file << "\n";
    linesCount++;
    // Close the file
    file.close();
}
```

I created a loop in code to test the different detectors and saved the results in a [CSV file](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/Output.csv).

MidTermProject_Camera_Student.cpp
```c++
std::vector<std::string> detectorTypeList = { "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
  
// create variables to csv file 
int det_keyponts [detectorTypeList.size()][imgEndIndex+1];
  
/* MAIN LOOP OVER ALL DETECTORS*/
for (size_t detIndex = 0; detIndex < detectorTypeList.size(); detIndex++)
{ 
    string  detectorType = detectorTypeList[detIndex];
    ...
}

// Creating an object of CSVWriter
CSVWriter writer(FileName);

// Creating a vector of strings
std::vector<std::string> header = { "Detector type", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
 
// Adding vector to CSV File
writer.addDatainRow(header.begin(), header.end());
for (size_t detIndex = 0; detIndex < detectorTypeList.size(); detIndex++)
{ 
    std::vector<std::string> dataList; 
    dataList.push_back(detectorTypeList[detIndex]);
    for (int i = 0; i < 10; i++)
        dataList.push_back(std::to_string(det_keyponts[detIndex][i]));
    // Wrote number of detector keyponts to csv file.
    writer.addDatainRow(dataList.begin(), dataList.end());
}   
```
Below results are shown for test images.

DETECTOR  | IMAGE               | Number of keypoints
--------  | --------------------| -------------------
SHITOMASI | ![alt text](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/images/SHITOMASI.png) | 111-125
HARRIS    | ![alt text](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/images/HARRIS.png)    | 14-43
FAST      | ![alt text](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/images/FAST.png)      | 138-156
BRISK     | ![alt text](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/images/BRISK.png)     | 254-297
ORB       | ![alt text](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/images/ORB.png)       | 92-130
AKAZE     | ![alt text](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/images/AKAZE.png)     | 155-179
SIFT      | ![alt text](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/images/SIFT.png)      | 124-159

SIFT and ORB detectors have small amount of groups of keypoints. Other detectors (SHITOMASI, HARRIS, FATS, BRISK, AKAZE) have big amount of groups of keypoints or separate keypoints.

#### 8. Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

I created a loop in code to test all possible combinations of detectors and descriptors and saved the results in a [CSV file](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/Matching.csv).

MidTermProject_Camera_Student.cpp
```c++
std::string FileName2 = "/home/workspace/SFND_2D_Feature_Matching/Matching.csv";

std::vector<std::string> descriptorTypeList = {"BRISK",  "BRIEF", "ORB", "FREAK"};

int match_keyponts[detectorTypeList.size()][descriptorTypeList.size()][imgEndIndex-1];

for (size_t desIndex = 0; desIndex < descriptorTypeList.size(); desIndex++)
{  
    string  descriptorType  = descriptorTypeList[desIndex];
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    ...
}

// Creating an object of CSVWriter
CSVWriter writer2(FileName2);

// Creating a vector of strings
std::vector<std::string> header2 = { "Detector+Descriptor type", "1", "2", "3", "4", "5", "6", "7", "8"};
// Adding vector to CSV File
writer2.addDatainRow(header2.begin(), header2.end());
for (size_t detIndex = 0; detIndex < detectorTypeList.size(); detIndex++)
{ 
    for (size_t desIndex = 0; desIndex < descriptorTypeList.size(); desIndex++)
    { 
        std::vector<std::string> dataList2; 
        std::string det_des_str = detectorTypeList[detIndex] + "/" + descriptorTypeList[desIndex];                     
        dataList2.push_back(det_des_str);
           for (int i = 2; i < 10; i++)
               dataList2.push_back(std::to_string(match_keyponts[detIndex][desIndex][i]));
        // Wrote number of detector keyponts to csv file.
        writer2.addDatainRow(dataList2.begin(), dataList2.end());
    }
}   
```
#### 9. Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.

I created a loop in code to test all possible combinations of detectors and descriptors and saved the results in a [CSV file](https://github.com/RustemIskuzhin/SFND-2D-Feature-Tracking/blob/master/Time.csv).

MidTermProject_Camera_Student.cpp
```c++
std::string FileName3 = "/home/workspace/SFND_2D_Feature_Matching/Time.csv";

int time_match_keyponts[detectorTypeList.size()][descriptorTypeList.size()][imgEndIndex+1];

time_match_keyponts[detIndex][desIndex][imgIndex] = (t1 + t2) * 1000.0;

// Creating an object of CSVWriter
CSVWriter writer3(FileName3);

// Creating a vector of strings
std::vector<std::string> header3 = { "Detector+Descriptor type", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
// Adding vector to CSV File
writer3.addDatainRow(header3.begin(), header3.end());
for (size_t detIndex = 0; detIndex < detectorTypeList.size(); detIndex++)
{ 
    for (size_t desIndex = 0; desIndex < descriptorTypeList.size(); desIndex++)
    { 
        std::vector<std::string> dataList3; 
        std::string det_des_str = detectorTypeList[detIndex] + "/" + descriptorTypeList[desIndex];                     
        dataList3.push_back(det_des_str);
        for (int i = 0; i < 10; i++)
            dataList3.push_back(std::to_string(time_match_keyponts[detIndex][desIndex][i]));
        // Wrote number of detector keyponts to csv file.
        writer3.addDatainRow(dataList3.begin(), dataList3.end());
    }
}   
```

The TOP3 detector / descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles are:
DETECTOR/DESCRIPTOR  | NUMBER OF KEYPOINTS | TIME
-------------------- | --------------------| -------------------
FAST+BRIEF           | 117 keypoints       | 1,4 ms
FAST+ORB             | 119 keypoints       | 1,7 ms
FAST+BRISK           | 80 keypoints        | 2 ms 
