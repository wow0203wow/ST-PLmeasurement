#include <jni.h>
#include "cn_edu_scau_leafcolor_ResultActivity.h"
#include "cn_edu_scau_leafcolor_ResultActivity_CalcTask.h"
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <queue>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <stdio.h>
#include <numeric>

#ifndef LOG_TAG
#define LOG_TAG "LEAF_COLOR_JNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) 
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) 
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) 
#endif

using namespace cv;
using namespace std;


typedef struct StruSeedlingInfo
{
    float stemDiameter;     
    float endpointNumber;   
    float seedlingHeight;   
    float processingTime;    
}Seedling_info;
Seedling_info  Info_tmp;
vector<Seedling_info> SeedlingInfoSave;    




typedef struct NodeX
{
    int x, y;
    int f, g, h;
    NodeX *father;
    NodeX(int xx, int yy)
    {
        this->x = xx;
        this->y = yy;
        this->f = 0;
        this->g = 0;
        this->h = 0;
        this->father = NULL;
    };
}NODE;

class AStar
{
public:
    AStar();
    ~AStar();

public:
    cv::Mat main_path;   
    void set_mapPath(cv::Mat map_path);
    void set_startPoint(int x, int y);
    void set_endPoint(int x, int y);

    bool search_path();

    void add_around();
    void add_around_singleDirection(int x, int y);

    bool check_node_mapRange(int x, int y);
    bool check_node_free(int x, int y);

    double compute_g(int x, int y);
    double compute_h(int x, int y);

    int check_openList(int x, int y);
    int check_closedList(int x, int y);
    void sort_openList();
    void quickSort(int left, int right, std::vector<NODE*>& arr);
    void draw_path(NODE *node);
    void clear_memory();

    //std::string m_map_path;   //20210718_LHW_原来的

    cv::Mat m_map_whole_color;
    cv::Mat m_map_whole_gray;

    int m_map_row, m_map_col;

    NODE* m_start_node;
    NODE* m_end_node;
    NODE* m_current_node;

    std::vector<NODE*> m_open_list;
    std::vector<NODE*> m_closed_list;

    double m_weight_x = 1;
    double m_weight_y = 1;
    double m_weight_diagonal = sqrt(2);
};

AStar::AStar()
{
}

AStar::~AStar()
{
}

double sum_d;

int float_to_int(float f)
{
    int *p = (int*)&f;
    int temp = *p;
    int sign = -1;
    if ((temp & 0x80000000) == 0)
    {
        sign = 1;
    }
    int exp;
    exp = (((temp >> 23) & 0x0ff)) - 127;
    int tail;
    tail = ((temp & 0x007fffff) | 0x00800000);
    int res = (tail >> (23 - exp));
    return res * sign;
}

double getProportion(Mat _referenceImg, float _realArea)
{
    double p;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    vector<double> heightArray;
    vector<vector<Point>> DaimsContours;

    Mat DrawingD = Mat::zeros(_referenceImg.size(), CV_8UC1);
    findContours(_referenceImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    
    vector<Rect> boundRect(contours.size());  
    vector<RotatedRect> box(contours.size()); 
    for (int i = 0; i<contours.size(); i++)
    {
        box[i] = minAreaRect(Mat(contours[i]));  
        boundRect[i] = boundingRect(Mat(contours[i]));
        heightArray.push_back(boundRect[i].height);       
        circle(_referenceImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  
        rectangle(_referenceImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(100, 255, 50), 2, 8);
    }
  
    sort(heightArray.begin(), heightArray.end());     
    float _pixelArea = 0;
 
    for (int i = 0; i < contours.size(); i++)
    {
        if (boundRect[0].height == heightArray[0])
        {
            DaimsContours.push_back(contours[0]);
        }
        drawContours(DrawingD, DaimsContours, 0, Scalar::all(255), CV_FILLED, 8, hierarcy, 0, Point()); 
        _pixelArea = fabs(contourArea(contours[0]));
        p = _realArea / _pixelArea;
    }
    return p;
}


Mat getMeasurementGoal(Mat pSrc, Mat pDst)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    vector<double> heightArray;
    vector<vector<Point>> DaimsContours;

    findContours(pSrc, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


    cout << "num=" << contours.size() << endl;
    vector<Rect> boundRect(contours.size());  

    for (int i = 0; i<contours.size(); i++)
    {
        boundRect[i] = boundingRect(Mat(contours[i]));
        heightArray.push_back(abs(boundRect[i].height));
    }   
    sort(heightArray.begin(), heightArray.end());  
    cout << "heightArray[0]=" << heightArray[0] << endl;
    
    for (int i = 0; i < contours.size(); i++)
    {
        if (boundRect[1].height == heightArray[1])
        {
            DaimsContours.push_back(contours[1]);
        }       
        Mat pDst0 = Mat::zeros(pSrc.size(), CV_8UC1);
        drawContours(pDst0, DaimsContours, 0, Scalar::all(255), CV_FILLED, 8, hierarcy, 0, Point()); 

        threshold(pDst0, pDst, 255, 255, CV_THRESH_OTSU);
        return pDst;        
    }
}

static int threshold_Lvalue = 0;
static int threshold_avalue = 0;
static int threshold_bvalue = 0;
static int threshold_LMax = 255;
static int threshold_aMax = 121;
static int threshold_bMax = 255;

Mat src, Lab, imgLabMask, Grayimg, filter_smallareaImg, thinImg;

void Delete_smallregions(Mat & pSrc, Mat & pDst)
{
    int size = 30;
    vector<vector<Point>> contours;           
    vector<Vec4i> hierarchy;                  
    findContours(pSrc, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    vector<vector<Point>>::iterator k;                   

    for (k = contours.begin(); k != contours.end();)     
    {
        if (contourArea(*k, false) < size)
        {
            k = contours.erase(k);
        }
        else
            ++k;
    }
   
    for (int i = 0; i < contours.size(); i++)
    {
        for (int j = 0; j < contours[i].size(); j++)
        {
            Point P = Point(contours[i][j].x, contours[i][j].y);
        }
        drawContours(pDst, contours, i, Scalar(255), -1, 8);
    }
}


Mat GetSeedlingBin(Mat srcImage, int threshold_Lvalue, int threshold_avalue, int threshold_bvalue)
{
    Mat dst_Lab;   
    vector<Mat> planes;    
    cvtColor(srcImage, dst_Lab, CV_RGB2Lab);
    split(Lab, planes);   
    Mat BW = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC1);
    for (int i = 0; i<srcImage.rows; i++)
    {
        for (int j = 0; j<srcImage.cols; j++)
        {
            if (((dst_Lab.at<Vec3b>(i, j)[0] >= threshold_Lvalue) && (dst_Lab.at<Vec3b>(i, j)[0] <= threshold_LMax)) &&
                ((dst_Lab.at<Vec3b>(i, j)[1] >= threshold_avalue) && (dst_Lab.at<Vec3b>(i, j)[1] <= threshold_aMax)) &&
                ((dst_Lab.at<Vec3b>(i, j)[2] >= threshold_bvalue) && (dst_Lab.at<Vec3b>(i, j)[2] <= threshold_bMax))
                    )
            {
                BW.at<uchar>(i, j) = 255;
            }
        }
    }
    threshold(BW, imgLabMask, 255, 255, CV_THRESH_OTSU);    
    filter_smallareaImg = Mat::zeros(imgLabMask.size(), CV_8UC1);
    Delete_smallregions(imgLabMask, filter_smallareaImg);
    Mat lvbo;    
    return filter_smallareaImg;  
}

void thinningIteration(cv::Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);   
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous())  
    {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;   
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    
    uchar *pDst;
  
    pAbove = NULL;
    pCurr = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (y = 1; y < img.rows - 1; ++y)
    {
        pAbove = pCurr;
        pCurr = pBelow;
        pBelow = img.ptr<uchar>(y + 1);
        pDst = marker.ptr<uchar>(y);
       
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols - 1; ++x)
        {          
            nw = no;
            no = ne;
            ne = &(pAbove[x + 1]);
            we = me;
            me = ea;
            ea = &(pCurr[x + 1]);
            sw = so;
            so = se;
            se = &(pBelow[x + 1]);

            int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                    (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                    (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                    (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }
    img &= ~marker;
}

void thinning(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();
    dst /= 255;       

    Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);  
        absdiff(dst, prev, diff);   
        dst.copyTo(prev);
    } while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

vector<Point> getEndpoints(Mat &thinSrc)
{
    int count;   
    uchar pix;
    vector<Point> coords_endpointcombination;    
    for (int i = 1; i < thinSrc.rows - 1; i++)  
    {
        for (int j = 1; j < thinSrc.cols - 1; j++)  
        {
            pix = thinSrc.at<uchar>(i, j);            
            if (pix == 0)
                continue;            
            count = 0;            
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {                    
                    pix = thinSrc.at<uchar>(i + y, j + x);                    
                    if (pix != 0)
                        count++;
                }
            }            
            if (count == 2)
            {
                coords_endpointcombination.push_back(Point(j, i)); 
            }
        }
    }
    return coords_endpointcombination;
}

void AStar::set_mapPath(cv::Mat map_path)
{   
    main_path = Mat::zeros(map_path.size(), CV_8UC1);  
    m_map_whole_color = map_path.clone();
    Mat m_map_whole_color_clone = map_path.clone();
    Mat imgGray;
    cvtColor(m_map_whole_color_clone, imgGray, CV_BGR2GRAY);
    m_map_whole_gray = imgGray;
    if (m_map_whole_gray.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
    }
    m_map_row = m_map_whole_gray.size().width;
    m_map_col = m_map_whole_gray.size().height;
}

void AStar::set_startPoint(int x, int y)
{
    m_start_node = new NodeX(x, y);
}

void AStar::set_endPoint(int x, int y)
{
    m_end_node = new NodeX(x, y);
}

bool AStar::search_path()
{
    if (m_start_node->x < 0 || m_start_node->x > m_map_row
        || m_start_node->y < 0 || m_start_node->y >m_map_col
        || m_end_node->x < 0 || m_end_node->x > m_map_row
        || m_end_node->y < 0 || m_end_node->y >m_map_col)
    {
        std::cout << "start/end point is out of the map range" << std::endl;
        return false;
    }
    m_current_node = m_start_node;
    m_open_list.push_back(m_start_node);
    while (m_open_list.size() > 0)
    {
        m_current_node = m_open_list[0];
        add_around();
        m_closed_list.push_back(m_current_node);
        m_open_list.erase(m_open_list.begin());
        sort_openList();
        if (m_current_node->x == m_end_node->x && m_current_node->y == m_end_node->y)
        {
            draw_path(m_current_node);
            clear_memory();
            std::cout << "Find path successful" << std::endl;
            return true;
        }
    }
    return false;
}

void AStar::add_around()
{
    add_around_singleDirection(m_current_node->x - 1, m_current_node->y - 1); // left up corner
    add_around_singleDirection(m_current_node->x, m_current_node->y - 1); // up
    add_around_singleDirection(m_current_node->x + 1, m_current_node->y - 1); // right up corner
    add_around_singleDirection(m_current_node->x - 1, m_current_node->y); // left
    add_around_singleDirection(m_current_node->x + 1, m_current_node->y); // right
    add_around_singleDirection(m_current_node->x - 1, m_current_node->y + 1); // left down corner
    add_around_singleDirection(m_current_node->x, m_current_node->y + 1); // down
    add_around_singleDirection(m_current_node->x + 1, m_current_node->y + 1); // right down corner
}

void AStar::add_around_singleDirection(int x, int y)
{
    if (!check_node_mapRange(x, y))
        return;
    if (!check_node_free(x, y))
        return;
    if (check_closedList(x, y) != -1)
        return;

    double g_new = m_current_node->g + compute_g(x, y);

    int id = check_openList(x, y);

    if (id != -1)
    {
        NODE* node_old = m_open_list[id];
        if (g_new < node_old->g)
        {
            node_old->g = g_new;
            node_old->f = node_old->g + node_old->h;
            node_old->father = m_current_node;
        }
    }
    else
    {
        NODE* node_new = new NodeX(x, y);
        node_new->g = g_new;
        node_new->h = compute_h(x, y);
        node_new->f = node_new->g + node_new->h;
        node_new->father = m_current_node;
        m_open_list.push_back(node_new);
    }
}

int AStar::check_openList(int x, int y)
{
    int num = m_open_list.size();
    for (int i = 0; i < num; i++)
    {
        if (x == m_open_list[i]->x && y == m_open_list[i]->y)
            return i;
    }
    return -1;
}

int AStar::check_closedList(int x, int y)
{
    int num = m_closed_list.size();
    for (int i = 0; i < num; i++)
    {
        if (x == m_closed_list[i]->x && y == m_closed_list[i]->y)
            return i;
    }
    return -1;
}

bool AStar::check_node_mapRange(int x, int y)
{
    if (x < 0 || x > m_map_row || y < 0 || y > m_map_col)
    {
        return false;
    }
    return true;
}

bool AStar::check_node_free(int x, int y)
{
    int color = m_map_whole_gray.at<uchar>(y, x);
    if (color > 250)
        return true;
    else
        return false;
}

double AStar::compute_g(int x, int y)
{
    double g = 0;
    if (abs(x - m_current_node->x) + abs(y - m_current_node->y) == 2)
    {
        g = m_weight_diagonal;
    }
    if (abs(x - m_current_node->x) == 0)
    {
        g = m_weight_y;
    }
    if (abs(y - m_current_node->y) == 0)
    {
        g = m_weight_x;
    }
    return g;
}

double AStar::compute_h(int x, int y)
{
    double h = sqrt(pow(x - m_end_node->x, 2) + pow(y - m_end_node->y, 2));
    return h;
}


void AStar::sort_openList()
{
    std::vector<double> f_list;
    quickSort(0, m_open_list.size() - 1, m_open_list);
}

void AStar::quickSort(int left, int right, std::vector<NODE*>& arr)
{
    if (left >= right)
        return;
    int i, j;
    NODE* base = arr[left];
    i = left, j = right;
    base = arr[left];
    while (i < j)
    {
        while (arr[j]->f >= base->f && i < j)
            j--;
        while (arr[i]->f <= base->f && i < j)
            i++;
        if (i < j)
        {
            NODE* temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    arr[left] = arr[i];
    arr[i] = base;
    quickSort(left, i - 1, arr);
    quickSort(i + 1, right, arr);
}



void AStar::draw_path(NODE *node)
{

    if (node->father != NULL)
    {
        cv::circle(main_path, cv::Point((int)(node->x), (int)(node->y)), 0.5, cv::Scalar(255, 255, 255), -1);       
        cv::circle(m_map_whole_color, cv::Point((int)(node->x), (int)(node->y)), 1, cv::Scalar(0, 0, 255), -1);
        draw_path(node->father);
    }
    else
    {       
        float d;
        vector<float>Xscale;
        vector<float>Yscale;
        vector<float>Count_d;  
        for (int i = 1; i < main_path.rows - 1; i++)   
        {
            for (int j = 1; j < main_path.cols - 1; j++) 
            {
                uchar pix;               
                pix = main_path.at<uchar>(i, j);                
                if (pix == 0)
                    continue;
                else
                {
                    Xscale.push_back(j);
                    Yscale.push_back(i);
                }
            }
        }
        for (int i = 0; i < Xscale.size() - 1; i++)    
        {           
            d = sqrt(powf((Xscale[i + 1] - Xscale[i]), 2) + powf((Yscale[i + 1] - Yscale[i]), 2));
            Count_d.push_back(d);           
        }       
        sum_d=accumulate(Count_d.begin(), Count_d.end(), 0.0000000);
    }
}

void AStar::clear_memory()
{
}

double getStembaseROI(Mat ppSrc)
{    
    int upOffset = 30;
    float diameter=0;
    float whiteX = 0;  
    float whiteY = 0;     
    uchar pix;   
    for (int i = 1; i < ppSrc.rows - 1; i++)  
    {
        for (int j = 1; j < ppSrc.cols - 1; j++)  
        {            
            pix = ppSrc.at<uchar>(i, j);            
            if (pix == 0) continue;           
            if (pix != 0)
            {
                whiteX = j;
                whiteY = i;
                break;   
            }
        }
    }    
    Rect rect(whiteX - whiteX, whiteY - upOffset, ppSrc.cols - 1, (ppSrc.rows - 1 + upOffset - whiteY));
    Mat roi = Mat(ppSrc, rect);
    Rect roi_rect = Rect(whiteX - whiteX, whiteY - upOffset, ppSrc.cols - 1, (ppSrc.rows - 1 + upOffset - whiteY));   
    Mat newRoi_1 = Mat::zeros(ppSrc.size(), CV_8UC1);
    roi.copyTo(newRoi_1(roi_rect));
    Mat newRoi = newRoi_1.clone();    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    findContours(newRoi, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);   
    vector<RotatedRect> box(contours.size());
    Point2f rectArray[4];
    for (int i = 0; i<contours.size(); i++)
    {
        box[i] = minAreaRect(Mat(contours[i]));        
        diameter = box[i].size.width;		
    }    
    return diameter;
}

extern "C" JNIEXPORT jobject JNICALL Java_cn_edu_scau_leafcolor_ResultActivity_seedlingTraits
        (JNIEnv *env, jobject, jobject bitmap, jdouble area){
    AndroidBitmapInfo info;
    void *pixels;

    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);


    Mat temp(info.height, info.width, CV_8UC4, pixels);
    Mat g;

    double  start = double(getTickCount());
    Mat src;
    cvtColor(temp, src, CV_RGBA2RGB);
    Mat imageSeedingBin0;
    resize(src, imageSeedingBin0, Size(640, 480));
    Mat imageSeedingBin = GetSeedlingBin(imageSeedingBin0, threshold_Lvalue, threshold_avalue, threshold_bvalue); 

    Mat imageSeedingBin_forHeight=imageSeedingBin.clone(); 
    resize(imageSeedingBin, g, temp.size());
    cvtColor(g, temp, CV_GRAY2RGBA);    

    Mat imageSeedingBin_getP = imageSeedingBin.clone();
    double p = getProportion(imageSeedingBin_getP, area);

    Mat GoalImg = Mat::zeros(imageSeedingBin.size(), CV_8UC1); 
    Mat out = getMeasurementGoal(imageSeedingBin, GoalImg);
    for (int i = 1; i < GoalImg.rows - 1; i++)   
    {
        for (int j = 1; j < GoalImg.cols - 1; j++) 
        {           
            int pix = GoalImg.at<uchar>(i, j);           
            if (pix == 0) continue;          
            if (pix != 0)
            {
                LOGI("此处非0");           
            }
        }
    }
    Mat GoalImg_to_stemwidth;
    GoalImg.copyTo(GoalImg_to_stemwidth);    
    thinning(GoalImg, thinImg);
    Mat getnum_endpointImg = thinImg.clone();   
    float stemwidth;
    stemwidth = getStembaseROI(GoalImg_to_stemwidth);    
    float stemDiameter_output;
    stemDiameter_output=(float_to_int(stemwidth*p*100))*1.0/100;
    Info_tmp.stemDiameter = stemDiameter_output;  

    vector<Point> coords_endpoints = getEndpoints(getnum_endpointImg);
    vector<Point> endpoints_for_Height = coords_endpoints;  
   
    int leafNumber_Output;
    leafNumber_Output = float_to_int(coords_endpoints.size()-1);
    Info_tmp.endpointNumber = leafNumber_Output;

    Point startPoint;
    Point endPoint;
   
    endpoints_for_Height.erase(endpoints_for_Height.begin()+1, endpoints_for_Height.end()-1);
    startPoint = endpoints_for_Height[0];
    endPoint = endpoints_for_Height[1];
   
    Mat threeChannel_Img = Mat::zeros(imageSeedingBin.rows, imageSeedingBin.cols, CV_8UC3);
    vector<Mat> channels;
    for (int i = 0; i<3; i++)
    {
        channels.push_back(imageSeedingBin_forHeight);
    }
    merge(channels, imageSeedingBin_forHeight);
    AStar astar;
    astar.set_mapPath(imageSeedingBin_forHeight);
    astar.set_startPoint(startPoint.x, startPoint.y);
    astar.set_endPoint(endPoint.x, endPoint.y);
    astar.search_path();
    double SUM_d;
    SUM_d=sum_d*p;
    
    float SUM_d_Putout;
    SUM_d_Putout = (float_to_int((SUM_d)*100))*1.0/100;
    Info_tmp.seedlingHeight = SUM_d_Putout;
   
    double  duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    cout << "duration_ms=" << duration_ms << "\n";
    
    float duration_ms_output;
    duration_ms_output=(float_to_int(duration_ms*100))*1.0/100;
    Info_tmp.processingTime = duration_ms_output;

    SeedlingInfoSave.push_back(Info_tmp);   
    AndroidBitmap_unlockPixels(env, bitmap);

    jclass rstClass = env->FindClass("cn/edu/scau/leafcolor/StruSeedlingInfo");
    jmethodID initFunID = env->GetMethodID(rstClass, "<init>", "()V");
    jobject result = env->NewObject(rstClass, initFunID);

    jfieldID tmp  = env->GetFieldID(rstClass, "endpointNumber", "F");
    env->SetFloatField(result, tmp, Info_tmp.endpointNumber);

    tmp  = env->GetFieldID(rstClass, "seedlingHeight", "F");
    env->SetFloatField(result, tmp, Info_tmp.seedlingHeight);

    tmp  = env->GetFieldID(rstClass, "stemDiameter", "F");
    env->SetFloatField(result, tmp, Info_tmp.stemDiameter);

    tmp  = env->GetFieldID(rstClass, "processingTime", "F");
    env->SetFloatField(result, tmp, Info_tmp.processingTime);

    return result;

}
