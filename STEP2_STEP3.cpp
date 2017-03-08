#include "opencv2/opencv.hpp"
#include <fstream>
#define MIN(a,b) ((a) < (b) ? (a) : (b))
using namespace cv;
using namespace std;
ofstream myfile;
const int somathreshold = 2000;
const int bluecontourthreshold = 1000;
RNG rng;
void arounddendrite(Mat imm, vector<RotatedRect> rotRect, ofstream &myfile, int CountLW, int CountMW, int CountHW)
{
 int LWlcount = 0, LWmcount = 0, LWhcount = 0, MWlcount = 0, MWmcount = 0, MWhcount = 0, HWlcount = 0, HWmcount = 0, HWhcount = 0;
 int totLWlcount = 0, totLWmcount = 0, totLWhcount = 0, totMWlcount = 0, totMWmcount = 0, totMWhcount = 0, totHWlcount = 0, totHWmcount = 0, totHWhcount = 0;
 int avgLWlcount = 0, avgLWmcount = 0, avgLWhcount = 0, avgMWlcount = 0, avgMWmcount = 0, avgMWhcount = 0, avgHWlcount = 0, avgHWmcount = 0, avgHWhcount = 0;
 Mat thr_high_Synapse;
 cv::inRange(imm, cv::Scalar(0, 0, 110), cv::Scalar(30, 30, 255), thr_high_Synapse);//RED-HIGH INTENSITY
 dilate(thr_high_Synapse, thr_high_Synapse, Mat());
 //imshow("thr_high_Synapse", thr_high_Synapse);
 Mat Medium_Synapse_thr;
 cv::inRange(imm, cv::Scalar(0, 0, 50), cv::Scalar(10, 10, 175), Medium_Synapse_thr);//RED -MED INTENSITY
 dilate(Medium_Synapse_thr, Medium_Synapse_thr, Mat());
 Mat Low_Synapse_thr;
 cv::inRange(imm, cv::Scalar(0, 0, 20), cv::Scalar(10, 10, 38), Low_Synapse_thr);
 Low_Synapse_thr = Low_Synapse_thr - Medium_Synapse_thr - thr_high_Synapse;//RED -LOW INTENSITY
 dilate(Low_Synapse_thr, Low_Synapse_thr, Mat());
 int totlowIntSyn = countNonZero(Low_Synapse_thr);
 int totmedIntSyn = countNonZero(Medium_Synapse_thr);
 int tothigIntSyn = countNonZero(thr_high_Synapse);
 for (int i = 0; i < rotRect.size(); i++)
 {
  LWlcount = 0, LWmcount = 0, LWhcount = 0, MWlcount = 0, MWmcount = 0, MWhcount = 0, HWlcount = 0, HWmcount = 0, HWhcount = 0;
  Mat Hrotated, Mrotated, Lrotated;
  cv::Mat Hcropped, Mcropped, Lcropped;
  cv::Mat rot_mat = cv::getRotationMatrix2D(rotRect[i].center, rotRect[i].angle, 1);
  if (rotRect[i].angle < -45.)
   std::swap(rotRect[i].size.width, rotRect[i].size.height);
  warpAffine(thr_high_Synapse, Hrotated, rot_mat, thr_high_Synapse.size(), INTER_CUBIC);
  // crop the resulting image
  cv::getRectSubPix(Hrotated, rotRect[i].size, rotRect[i].center, Hcropped);
  warpAffine(Medium_Synapse_thr, Mrotated, rot_mat, Medium_Synapse_thr.size(), INTER_CUBIC);
  // crop the resulting image
  cv::getRectSubPix(Mrotated, rotRect[i].size, rotRect[i].center, Mcropped);
  warpAffine(Medium_Synapse_thr, Lrotated, rot_mat, Low_Synapse_thr.size(), INTER_CUBIC);
  // crop the resulting image
  cv::getRectSubPix(Lrotated, rotRect[i].size, rotRect[i].center, Lcropped);// low int synapse rotated rect (ROI)  image
  float wdth = MIN(rotRect[i].size.width, rotRect[i].size.height);
  if (wdth >= 1)
  {
   if (wdth <= 50) //low width dendrite
   {
    LWlcount = countNonZero(Lcropped); // no of low int syn around a Low width dendrite
    LWmcount = countNonZero(Mcropped);// no of med int syn around a Low width dendrite
    LWhcount = countNonZero(Hcropped);// no of high int syn around a Low width dendrite
    //cout << countNonZero(Lcropped) << ", " << countNonZero(Mcropped) << ",  " << countNonZero(Hcropped) << endl;
    totLWlcount += LWlcount; totLWmcount += LWmcount;  totLWhcount += LWhcount;
   }
   else if (wdth > 50 && wdth <= 100)
   {
    MWlcount = countNonZero(Lcropped);
    MWmcount = countNonZero(Mcropped);
    MWhcount = countNonZero(Hcropped);
    totMWlcount += MWlcount; totMWmcount += MWmcount;  totMWhcount += MWhcount;
   }
   else
   {
    HWlcount = countNonZero(Lcropped);
    HWmcount = countNonZero(Mcropped);
    HWhcount = countNonZero(Hcropped);
    totHWlcount += HWlcount; totHWmcount += HWmcount;  totHWhcount += HWhcount;
   }
  }
 }
 if (CountLW == 0) { avgLWlcount = 0; avgLWmcount = 0; avgLWhcount = 0; }
 else { avgLWlcount = totLWlcount / CountLW;  avgLWmcount = totLWmcount / CountLW; avgLWhcount = totLWhcount / CountLW; }
 if (CountMW == 0) { avgMWlcount = 0; avgMWmcount = 0; avgMWhcount = 0; }
 else { avgMWlcount = totMWlcount / CountMW; avgMWmcount = totMWmcount / CountMW; avgMWhcount = totMWhcount / CountMW; }
 if (CountHW == 0) { avgHWlcount = 0; avgHWmcount = 0; avgHWhcount = 0; }
 else { avgHWlcount = totHWlcount / CountHW; avgHWmcount = totHWmcount / CountHW; avgHWhcount = totHWhcount / CountHW; }
 //myfile << "," << "arounddendrite" << "," << "Avg Low Int Syn count arnd. Low width" << "," << "Avg Med Int Syn count arnd. Low width" << "," << "Avg Low Int Syn count arnd. High width" << "," << "Avg Low Int Syn count arnd. Med width" << "," << " Avg Med Int Syn count arnd. Med width" << "," << "Avg High Int Syn count arnd. Med width" << "," << "Avg Low Int Syn count arnd. Large width " << "," << "Avg Med Int Syn count arnd. Large width" << "," << "Avg High Int Syn count arnd. Large width";
 myfile << "," << "arounddendrite" << "," << avgLWlcount << "," << avgLWmcount << "," << avgLWhcount << "," << avgMWlcount << "," << avgMWmcount << "," << avgMWhcount << "," << avgHWlcount << "," << avgHWmcount << "," << avgHWhcount;
}
void dendritedetect(Mat img, ofstream & myfile, string fstr, int n)
{
 //string dendrpath = format("C:\\Users\\VIneeta\\Documents\\Visual Studio 2013\\Projects\\OpenCV-Test\\OpenCV-Test\\Old STEP1\\%s\\den_z%d_3layers_processed.png", fstr.c_str(), n);
 string dendrpath  = format("C:\\Users\\VIneeta\\Pictures\\STEPi\\%s\\den_%s_z%d_processed.jpg", fstr.c_str(), fstr.c_str(), n);
 Mat imgtofun = img.clone();
 vector<vector<Point>> checkcontours;
 vector<Vec4i> checkhierarchy;
 Mat immg = img.clone();
 //cout << dendrpath << endl;
 cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
 Mat cloned = img.clone();
 Mat hthr;
 //cv::inRange(img, cv::Scalar(0, 17, 2), cv::Scalar(60, 255, 150), img);//Green-yellow THRESH
 cv::inRange(img, cv::Scalar(20, 0, 20), cv::Scalar(255, 100, 225), img);//HIGH PURPLE THRESH
 int Swid = 0, Mwid = 0, Lwid = 0, totSwid = 0, totMwid = 0, totLwid = 0, avgSwid = 0, avgMwid = 0, avgLwid = 0, avgSlen = 0, avgMlen = 0, avgLlen = 0;
 int totSlen = 0, totMlen = 0, totLlen = 0;
 Mat thresh = img.clone(); // original thresholded image of greeen dendrites
 int avgoverallwdth = 0, avgoverallen = 0;
 cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
 cv::Mat temp(img.size(), CV_8UC1);
 cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
 bool done;
 do
 {
  cv::morphologyEx(img, temp, cv::MORPH_OPEN, element);
  cv::bitwise_not(temp, temp);
  cv::bitwise_and(img, temp, temp);
  // cout << "img size  " << img.size() << endl << "temp.size  " << temp.size() << endl << " skel size  " << skel.size() << endl;
  cv::bitwise_or(skel, temp, skel);
  cv::erode(img, img, element);
  double max;
  cv::minMaxLoc(img, 0, &max);
  done = (max == 0);
 } while (!done);
 Mat cdst;
 cvtColor(skel, cdst, CV_GRAY2BGR);
 // imwrite("skeleton.png",skel);
 dilate(skel, skel, Mat());
 /** imwrite the image -------------------------------------------------------------------- Display purposes only*/
 vector<vector<Point>> dispcontours; vector<Vec4i> disphierarchy;
 findContours(skel, dispcontours, disphierarchy, CV_RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
 for (int i = 0; i < dispcontours.size(); i++)
 {
  if (arcLength(dispcontours[i], false)>250)
   drawContours(cloned, dispcontours, i, Scalar(20, 230, 240), 1, cv::LINE_8, vector<Vec4i>(), 0, Point());
 }
 imwrite(dendrpath, cloned);
 vector<vector<Point>> hcontours;
 vector<Vec4i> hhierarchy;
 dilate(skel, skel, Mat());
 findContours(skel, hcontours, hhierarchy, CV_RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
 vector<RotatedRect> minRect(hcontours.size());
 int wdth = 0, swidth = 0, mwidth = 0, lwidth = 0;
 int slen = 0, mlen = 0, llen = 0;
 for (int i = 0; i < hcontours.size(); i++)
 {
  minRect[i] = minAreaRect(Mat(hcontours[i]));
 }
 for (int i = 0; i< hcontours.size(); i++)
 {
  if ((arcLength(hcontours[i], true) >= 250) && (hcontours[i].size() >= 5))
  {
   Size2f s = minRect[i].size;
   wdth = MIN(s.width, s.height);
   if (wdth >= 1)
   {
    if (wdth <= 50)
    {
     totSwid += wdth;
     swidth++;
    }
    else if (wdth > 50 && wdth <= 100)
    {
     totMwid += wdth;
     mwidth++;
    }
    else
    {
     totLwid += wdth;
     lwidth++;
    }
   }
   if ((arcLength(hcontours[i], true)) <= 350)
   {
    totSlen += (arcLength(hcontours[i], true));
    slen++;
   }
   else if ((arcLength(hcontours[i], true) >= 350) && (arcLength(hcontours[i], true) <= 600))
   {
    totMlen += (arcLength(hcontours[i], true));
    mlen++;
   }
   else
   {
    totLlen += (arcLength(hcontours[i], true));
    llen++;
   }
  }
 }
 if (slen == 0) avgSlen = 0;
 else avgSlen = totSlen / slen;
 if (mlen == 0) avgMlen = 0;
 else avgMlen = totMlen / mlen;
 if (llen == 0) avgLlen = 0;
 else  avgLlen = totLlen / llen;
 if (swidth == 0) avgSwid = 0;
 else avgSwid = totSwid / swidth;
 if (mwidth == 0) avgMwid = 0;
 else avgMwid = totMwid / mwidth;
 if (lwidth == 0) avgLwid = 0;
 else avgLwid = totLwid / lwidth;
 if ((swidth + mwidth + lwidth) == 0) avgoverallwdth = 0;
 else avgoverallwdth = (totSwid + totMwid + totLwid) / (swidth + mwidth + lwidth);
 if ((slen + mlen + llen) == 0)avgoverallen = 0;
 else avgoverallen = (totSlen + totMlen + totLlen) / (slen + mlen + llen);
 /*~~~~~~~~~~~~~~*/arounddendrite(imgtofun, minRect, myfile, swidth, mwidth, lwidth);
 //myfile << "," << "DENDRITE" << "," << "Total No of Dendrites" << "," << "No.of Small width Dendrites" << "," << " No.of med width Dendrites" << "," << "No.of large width Dendrites" << "," << "No.of Small length Dendrites" << "," << "No.of Med length Dendrites" << "," << "No.of Large length Dendrites" << "," << "Avg len of all Dendrites" << "," << "Avg len of small length Dendrites" << "," << "Avg len of med length Dendrites" << "," << "Avg len of large length Dendrites" << "," << "Avg wdth of all Dendrites" << "," << "Avg wdth of small width Dendrites" << "," << "Avg wdth of med width Dendrites" << "," << "Avg wdth of large width Dendrites" << ",";
 myfile << "," << "DENDRITE" << "," << swidth + mwidth + lwidth << "," << swidth << "," << mwidth << "," << lwidth << "," << slen << "," << mlen << "," << llen << "," << avgoverallen << "," << avgSlen << "," << avgMlen << "," << avgLlen << "," << avgoverallwdth << "," << avgSwid << "," << avgMwid << "," << avgLwid << ",";
}
void findsoma(Mat &im, vector<vector<Point>> blue_contours, vector<vector<Point>> intersection_contours, ofstream &myfile)
{
 vector<vector<Point>> astrocytes;
 vector<Vec4i> astrocyteshierarchy;
 vector<vector<Point>> neurons;
 vector<Vec4i> neuronshierarchy;
 vector<vector<Point>> final_soma_contours;
 vector<Vec4i> final_soma_hierarchy;
 vector<vector<Point>> refine_intersect_contours;
 vector<Vec4i> afhierarchy;
 vector<vector<Point>> refine_blue_contours;
 vector<Vec4i> bfhierarchy;
 vector<vector<Point>> scontours;
 vector<Vec4i> shierarchy;
 int soma_indices[100];
 vector<vector<Point>> remcontours;
 vector<Moments> moment_blue(blue_contours.size());
 vector<Point2f> masscenter_blue(blue_contours.size());
 int ib = -1;
 int count = 0;
 vector<Point> som;
 vector<Moments> moment_intersect(intersection_contours.size());
 vector<Point2f> masscenter_intersect(intersection_contours.size());
 int i1 = -1;
 for (int i = 0; i < intersection_contours.size(); i++) // filter areas off intersection contour
 {
  if (fabs(contourArea(intersection_contours[i])) > somathreshold)
   refine_intersect_contours.push_back(intersection_contours[i]);
 }
 for (int i = 0; i < blue_contours.size(); i++) // filter areas of blue contours
 {
  if (fabs(contourArea(blue_contours[i])) > bluecontourthreshold)
   refine_blue_contours.push_back(blue_contours[i]);
 }
 /// Get the moments of blue contours
 for (int i = 0; i < refine_blue_contours.size(); i++)
  moment_blue[i] = moments(refine_blue_contours[i], false);
 ///  Get the mass centers of blue contours:
 for (int i = 0; i < refine_blue_contours.size(); i++)
  masscenter_blue[i] = Point2f(moment_blue[i].m10 / moment_blue[i].m00, moment_blue[i].m01 / moment_blue[i].m00);
 /// Get the moments of intersection contours
 for (int i = 0; i < refine_intersect_contours.size(); i++)
  moment_intersect[i] = moments(refine_intersect_contours[i], false);
 ///  Get the mass centers of intersection contours:
 for (int i = 0; i < refine_intersect_contours.size(); i++)
  masscenter_intersect[i] = Point2f(moment_intersect[i].m10 / moment_intersect[i].m00, moment_intersect[i].m01 / moment_intersect[i].m00);
 // check if centroid of any of the intersection contours is near the blue contours: If yes then- we have found a soma
 float d = 0; int j1 = 0; int ifin = -1; int j5 = -1; int jfin = 0;
 if (refine_blue_contours.size() == 0)
 {
  myfile << "," << "TOTAL" << "," << 0;
  myfile << "," << "SOMA" << " ," << 0 << " ," << 0 << " ," << 0 << ", " << 0 << " ," << 0;
  myfile << "," << "Astrocyte" << " ," << 0 << " ," << 0 << " ," << 0;
  myfile << "," << "Others" << " ," << 0 << " ," << 0 << " ," << 0;
 }
 else
 {
  for (int i = 0; i < refine_intersect_contours.size(); i++)
  {
   som.clear();
   float dmin = 1000000; int index = 0;
   for (int j = 0; j < refine_blue_contours.size(); j++)
   {
    d = sqrt(((masscenter_intersect[i].x - masscenter_blue[j].x)*(masscenter_intersect[i].x - masscenter_blue[j].x)) + ((masscenter_intersect[i].y - masscenter_blue[j].y)*(masscenter_intersect[i].y - masscenter_blue[j].y)));
    if (d < dmin)
    {
     dmin = d;
     index = j;
    }
   }
   final_soma_contours.push_back(refine_blue_contours[index]);
   soma_indices[i] = index;
  }
  // loop to find remaining contours
  for (int g = 0; g < refine_blue_contours.size(); g++)
  {
   int cc = 0;
   for (int h = 0; h < 100; h++)
   {
    if (g == soma_indices[h])// depicts a soma
     cc++;
   }
   if (cc == 0)
    remcontours.push_back(refine_blue_contours[g]); // not a soma
  }
  int alowcirc = 0, ahighcirc = 0;// circularity of astrocytes
  int nlowcirc = 0, nhighcirc = 0;// circularity of neural cells
  for (int k = 0; k < remcontours.size(); k++)
  {
   cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(remcontours[k]));
   float aspect_ratio = float(min_area_rect.size.width) / min_area_rect.size.height;
   if (aspect_ratio > 1.0) {
    aspect_ratio = 1.0 / aspect_ratio;
   }
   if (aspect_ratio >= 0.35 && aspect_ratio <= 0.7)
   {
    astrocytes.push_back(remcontours[k]);
    if (aspect_ratio <= 0.55)
     alowcirc++;
    else
     ahighcirc++;
   }
   else
   {
    neurons.push_back(remcontours[k]);
    if (aspect_ratio <= 0.55)
     nlowcirc++;
    else
     nhighcirc++;
   }
  }
  int siz;
  vector<Point2f>center(final_soma_contours.size());
  vector<float>radius(final_soma_contours.size());
  vector<float> aspect_ratio(final_soma_contours.size());
  vector<float> diameter;
  float sumaspect = 0;
  float meanaspect = 0;
  float stddevaspect = 0;
  float sumdiameter = 0;
  float meandiameter = 0;
  float stddevdiameter = 0;
  siz = final_soma_contours.size();
  for (int k = 0; k < final_soma_contours.size(); k++)
  {
   minEnclosingCircle((Mat)final_soma_contours[k], center[k], radius[k]);
   circle(im, center[k], radius[k], Scalar(20, 230, 125), 6, 8, 0);
   cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(final_soma_contours[k]));
   aspect_ratio[k] = float(min_area_rect.size.width) / min_area_rect.size.height;
   if (aspect_ratio[k] > 1.0)
    aspect_ratio[k] = 1.0 / aspect_ratio[k];
  }
  for (int k = 0; k < final_soma_contours.size(); k++)
  {
   sumaspect += aspect_ratio[k];
   sumdiameter += (2 * radius[k]);
  }
  meanaspect = sumaspect / siz;
  meandiameter = sumdiameter / siz;
  for (int k = 0; k < final_soma_contours.size(); k++)
  {
   stddevaspect = ((meanaspect - aspect_ratio[k])*(meanaspect - aspect_ratio[k])) / siz;
   stddevdiameter = ((meandiameter - (2 * radius[k]))*(meandiameter - (2 * radius[k]))) / siz;
  }
  myfile << "," << "TOTAL" << "," << refine_blue_contours.size();
  myfile << "," << "SOMA" << " ," << final_soma_contours.size() << " ," << meandiameter << " ," << stddevdiameter << ", " << meanaspect << " ," << stddevaspect;
  myfile << "," << "Astrocyte" << " ," << astrocytes.size() << " ," << alowcirc << " ," << ahighcirc;
  myfile << "," << "Others" << " ," << neurons.size() << " ," << nlowcirc << " ," << nhighcirc;
  for (int j = 0; j < astrocytes.size(); j++)
   drawContours(im, astrocytes, j, Scalar(255, 255, 255), 2, cv::LINE_8, vector<Vec4i>(), 0, Point());
  for (int j = 0; j < neurons.size(); j++)
   drawContours(im, neurons, j, Scalar(255, 255, 0), 2, cv::LINE_8, vector<Vec4i>(), 0, Point());
  //cout << "Soma Size : = " << siz << endl;
  imwrite("immmm.png", im);
 }
}
void areabinning(vector<vector<Point>> contours, string str, ofstream &myfile)
{
 int c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0;
 for (int i = 0; i < contours.size(); i++)
 {
  double area = fabs(contourArea((contours[i])));
  if (area >= 5 && area < 100)
   c1++;
  if (area >= 100 && area < 200)
   c2++;
  if (area >= 200 && area < 300)
   c3++;
  if (area >= 300 && area < 400)
   c4++;
  if (area >= 400 && area < 500)
   c5++;
  if (area >= 500)
   c6++;
 }
 myfile << " ," << "SYNAPSE" << "," << str << "," << c1 << "," << c2 << "," << c3 << "," << c4 << "," << c5 << "," << c6 << ",";
}
void synapse(Mat &imm, ofstream &myfile)
{
 vector<vector<Point>> High_synapse_contours;
 vector<Vec4i> thierarchy;
 Mat thr_high_Synapse;
 cv::inRange(imm, cv::Scalar(0, 0, 110), cv::Scalar(30, 30, 255), thr_high_Synapse);//RED-HIGH INTENSITY
 dilate(thr_high_Synapse, thr_high_Synapse, Mat());
 findContours(thr_high_Synapse, High_synapse_contours, thierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the contours in the image
 for (int i = 0; i < High_synapse_contours.size(); i++)
  drawContours(imm, High_synapse_contours, i, Scalar(128, 0, 255), cv::FILLED, cv::LINE_8, vector<Vec4i>(), 0, Point());//pink-high-purple
 /*~~~~~~~~~~~~*/areabinning(High_synapse_contours, "high intensity", myfile);
 // medium intensity
 vector<vector<Point>> contours;
 vector<Vec4i> hierarchy;
 Mat Medium_Synapse_thr;
 Mat thrdil;
 dilate(thr_high_Synapse, thrdil, Mat());
 cv::inRange(imm, cv::Scalar(0, 0, 50), cv::Scalar(10, 10, 175), Medium_Synapse_thr);//RED -LOW+HIGH INTENSITY
 Medium_Synapse_thr = Medium_Synapse_thr - thrdil;                // RED (LOW INTENSITY)
 findContours(Medium_Synapse_thr, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the contours in the image
 for (int i = 0; i < contours.size(); i++)
 {
  drawContours(imm, contours, i, Scalar(255, 255, 0), cv::FILLED, cv::LINE_8, vector<Vec4i>(), 0, Point());//green-medium-dark pink
 }
 /*~~~~~~~~~~~~*/areabinning(contours, "medium intensity", myfile);
 vector<vector<Point>> contoursm;
 vector<Vec4i> hierarchym; Mat Low_Synapse_thr;
 cv::inRange(imm, cv::Scalar(0, 0, 20), cv::Scalar(10, 10, 38), Low_Synapse_thr);
 Low_Synapse_thr = Low_Synapse_thr - Medium_Synapse_thr - thr_high_Synapse;
 //dilate(thrm, thrm, Mat());
 findContours(Low_Synapse_thr, contoursm, hierarchym, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the contours in the image
 for (int i = 0; i < contoursm.size(); i++)
 {
  drawContours(imm, contoursm, i, Scalar(255, 255, 255), cv::FILLED, cv::LINE_8, vector<Vec4i>(), 0, Point());//orange color-low
 }
 imwrite("Synapse_Show.png", imm);
 /*~~~~~~~~~~~~*/areabinning(contoursm, "low intensity", myfile);
}
vector<vector<Point>> watershedcontours(Mat src)
{
 int noofruns = 30;
 double min, max;
 Point maxLoc;
 vector<vector<Point>> contours_check, large_contours;
 Mat bw, bin;
 vector<Mat> storewatershed;
 //cv::inRange(src, cv::Scalar(20, 0, 0), cv::Scalar(255, 30, 60), bw);//BLUE THRESH
 //imshow("Binary Image", bw);
 //thresholding nuclei (blue) in the image
 Mat bgr[3];   //destination array
 split(src, bgr);//split source
 //Note: OpenCV uses BGR color order
 threshold(bgr[0], bw, 40, 255, THRESH_BINARY | THRESH_OTSU);// blue channel threshold
 //imshow("Binary Image", bw);
 Mat kernel_op = getStructuringElement(MORPH_RECT, Size(3, 3));
 Mat morph1;
 morphologyEx(bw, morph1, CV_MOP_OPEN, kernel_op);
 //imshow("openmorphology", morph1);
 Mat morph;
 Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
 morphologyEx(morph1, morph, CV_MOP_CLOSE, kernel2);
 //erode(morph, morph, Mat());
 //imshow("after morphology", morph);
 bw = morph;
 // Perform the distance transform algorithm
 Mat dist;
 distanceTransform(bw, dist, CV_DIST_L2, 3);
 normalize(dist, dist, 0, 1., NORM_MINMAX);
 //imshow("Distance Transform Image", dist);
 threshold(dist, dist, .2, 1., CV_THRESH_BINARY);
 // Dilate a bit the dist image
 Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
 dilate(dist, dist, kernel1);
 //imshow("Peaks", dist);
 //cout << countNonZero(dist) << endl;
 if (countNonZero(dist) > 70000)
  noofruns = 15;
 // Create the CV_8U version of the distance image
 // It is needed for findContours()
 Mat dist_8u, distback_8u;
 dist.convertTo(dist_8u, CV_8U);
 // Find total markers
 vector<vector<Point> > contours, backcontours;
 findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
 Mat dist_back;
 threshold(dist, dist_back, 0, 0.5, CV_THRESH_BINARY_INV);
 Mat kern_erod = getStructuringElement(MORPH_RECT, Size(5, 5));
 for (int y = 0; y < noofruns; y++)
 {
  erode(dist_back, dist_back, kern_erod);
 }
 //imshow("PeaksBackground", dist_back);
 dist_back.convertTo(distback_8u, CV_8U);
 findContours(distback_8u, backcontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
 Mat markers = Mat::zeros(dist.size(), CV_32SC1);
 // Draw the foreground markers
 for (size_t i = 0; i < contours.size(); i++)
 {
  drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
 }
 for (size_t i = 0; i < backcontours.size(); i++)
 {
  drawContours(markers, backcontours, (i), Scalar(255, 255, 255), -1);
 }
 int ncomp = contours.size();
 //imshow("Markers", markers * 10000);
 // Perform the watershed algorithm
 watershed(src, markers);
 vector<Vec3b> colors;
 for (size_t i = 0; i < contours.size(); i++)
 {
  int b = theRNG().uniform(0, 255);
  int g = theRNG().uniform(0, 255);
  int r = theRNG().uniform(0, 255);
  colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
 }
 vector<Mat> images2; vector<vector<Point> > final_contours, all_finalcontours;
 for (int obj = 1; obj < contours.size() + 1; obj++)
 {
  Mat dst2;
  src.copyTo(dst2, (markers == obj));
  cv::inRange(dst2, cv::Scalar(40, 0, 0), cv::Scalar(255, 55, 45), dst2);//BLUE THRESH
  morphologyEx(dst2, dst2, CV_MOP_CLOSE, kernel2);
  morphologyEx(dst2, dst2, CV_MOP_CLOSE, kernel2);
  findContours(dst2, final_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  for (int u = 0; u < final_contours.size(); u++)
   all_finalcontours.push_back(final_contours[u]);
  images2.push_back(dst2.clone());
 }
 for (size_t p = 0; p < all_finalcontours.size(); p++)
 {
  Scalar rnd_colors = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
  drawContours(src, all_finalcontours, p, rnd_colors, cv::FILLED, cv::LINE_8, vector<Vec4i>(), 0, Point());
 }
 //imshow("src", src);
 return all_finalcontours;
}
int main(int argc, char** argv)
{
 std::string file_contents;
 myfile.open("STEP2i_waste.csv");
 std::ifstream file("step.txt");
 std::string fstr; string name;
 vector<vector<Point>> blue_contours;
 vector<Vec4i> bhierarchy;
 vector<vector<Point>> intersect_contours;
 vector<vector<Point>> mod_intersect_contours;
 vector<Vec4i> ahierarchy;
 Mat bthr, src_gray, gth; Mat added; Mat intersection;
 using namespace cv;
 vector<vector<Point>> tcontours;
 vector<Vec4i> thierarchy;
 // Read image
  while (std::getline(file, fstr))
 {
 // Read image
 for (int n = 1; n <= 40; n++)
 {
 if (n < 10)
  name = format("C:\\Users\\VIneeta\\Pictures\\STEPi\\%s\\%s_z0%dc1+2+3.jpg", fstr.c_str(), fstr.c_str(), n);
 //name = format("C:\\Users\\VIneeta\\Documents\\Visual Studio 2013\\Projects\\OpenCV-Test\\OpenCV-Test\\Old STEP1i\\%s\\z%d_3layers_original.tif", fstr.c_str(), n);
 else
  name = format("C:\\Users\\VIneeta\\Pictures\\STEPi\\%s\\%s_z%dc1+2+3.jpg", fstr.c_str(), fstr.c_str(), n);
 //name = format("C:\\Users\\VIneeta\\Documents\\Visual Studio 2013\\Projects\\OpenCV-Test\\OpenCV-Test\\Old STEP1i\\%s\\z%d_3layers_original.tif", fstr.c_str(), n);
 cout << format("Processsing Folder %s", fstr.c_str()) << endl;
 Mat imm = imread(name);
 if (imm.empty()) //if image is empty- read next line of txt file- move on to the next folder
 { goto POP; cout << "NOT FOUND:"<< endl;}
 //string result = format("C:\\Users\\VIneeta\\Documents\\Visual Studio 2013\\Projects\\OpenCV-Test\\OpenCV-Test\\Old STEP1i\\%s\\z%d_3layers_processed.tif", fstr.c_str(), n);
 string result= format("C:\\Users\\VIneeta\\Pictures\\STEPi\\%s\\%s_z%dc1+2+3_processed.jpg", fstr.c_str(), fstr.c_str(), n);
 string imagnam = format("%s_z%dc1+2+3", fstr.c_str(), n);
 myfile << imagnam;
 cout << format("Processsing Image %s", imagnam.c_str()) << endl;
 Mat imtofn = imm.clone();
 cvtColor(imm, src_gray, CV_BGR2GRAY);
 /*~~~~~~~*/ synapse(imm, myfile);
 /*~~~~~~~*/dendritedetect(imtofn, myfile, fstr, n);
 // thresholding bright green and yellow (soma tails/dendrites)
 /*cv::threshold(src_gray, src_gray, 50, 255, cv::THRESH_TOZERO);
 bitwise_not(src_gray, src_gray);
 cv::GaussianBlur(src_gray, gth, cv::Size(3, 3), 0, 0);
 cv::threshold(gth, gth, 200, 255, cv::THRESH_BINARY);
 bitwise_not(gth, gth);*/
 cv::inRange(imm, cv::Scalar(20, 0, 20), cv::Scalar(255, 100, 225), gth);//HIGH PURPLE THRESH
 //dilate(gth, gth, Mat());
 //thresholding nuclei (blue) in the image
 /*Mat bgr[3];   //destination array
 split(imm, bgr);//split source
 //Note: OpenCV uses BGR color order
 threshold(bgr[0], bthr, 40, 200, THRESH_BINARY);// blue channel threshold*/
 cv::inRange(imm, cv::Scalar(40, 0, 0), cv::Scalar(255, 55, 45), bthr);//BLUE THRESH
 bitwise_and(gth, bthr, intersection);//intersection of nuclei with dendrite
 //findContours(bthr, blue_contours, bhierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the blue contours in the image
 findContours(intersection, intersect_contours, ahierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the contours in the intersection image
 /*~~~~~~~*/blue_contours = watershedcontours(imtofn);
 /*~~~~~~~*/ findsoma(imm, blue_contours, intersect_contours, myfile);
 //-------------------------------------------------------------------------------------------------------------------------------------------   
 imwrite(result, imm);
 myfile << endl;
  }
 POP:;
 }
 myfile.close();
 waitKey(0);
}
