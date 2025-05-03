#include "stdafx.h"
#include "common.h"
#include "OpenCVApplication.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;
using namespace cv;

// 1. Load CIFAR-10 binary batch
bool loadCIFAR10Batch(const string& filename,
    vector<Mat>& images,
    vector<int>& labels)
{
    constexpr int W = 32, H = 32, C = 3, SZ = W * H * C, N = 10000;
    ifstream file(filename, ios::binary);
    if (!file) { cerr << "Can't open " << filename << "\n"; return false; }
    images.clear(); labels.clear(); images.reserve(N); labels.reserve(N);
    vector<uint8_t> buf(SZ);
    for (int i = 0; i < N; ++i) {
        uint8_t l; file.read((char*)&l, 1); labels.push_back(l);
        file.read((char*)buf.data(), SZ);
        Mat img(H, W, CV_8UC3);
        for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c) {
            int idx = r * W + c;
            img.at<Vec3b>(r, c) = Vec3b(
                buf[2 * W * H + idx], buf[W * H + idx], buf[idx]
            );
        }
        images.push_back(img);
    }
    cout << "Loaded " << images.size() << " imgs from " << filename << "\n";
    return true;
}

// 2. Convert to grayscale
bool convertToGrayscale(const vector<Mat>& in,
    vector<Mat>& out)
{
    if (in.empty()) { cerr << "No images to convert\n"; return false; }
    out.clear(); out.reserve(in.size());
    for (auto& img : in) {
        Mat g; cvtColor(img, g, COLOR_BGR2GRAY);
        out.push_back(g);
    }
    cout << "Converted " << out.size() << " imgs to gray\n";
    return true;
}

// 3. Extract features
bool detectAndCompute(const vector<Mat>& gray,
    Ptr<Feature2D> det,
    vector<Mat>& descs)
{
    if (gray.empty()) { cerr << "No gray imgs\n"; return false; }
    descs.clear();
    for (auto& g : gray) {
        vector<KeyPoint> k;
        Mat d;
        det->detectAndCompute(g, noArray(), k, d);
        if (!d.empty()) descs.push_back(d);
    }
    cout << "Got descriptors for " << descs.size() << " imgs\n";
    return !descs.empty();
}

// 4. Build visual dictionary (k-means)
bool buildVocabulary(const vector<Mat>& descs,
    int dictSize,
    Mat& vocab)
{
    // TODO: Cluster descriptors into 'dictSize' words.
    return true;
}

// 5. Init BOW extractor
bool setupBowExtractor(Ptr<Feature2D> det,
    Ptr<DescriptorMatcher> matcher,
    const Mat& vocab,
    Ptr<BOWImgDescriptorExtractor>& bowExt)
{
    // TODO: Create BOWImgDescriptorExtractor and set vocab.
    return true;
}

// 6. Compute BoW histograms
bool computeBowHistograms(const vector<Mat>& gray,
    Ptr<Feature2D> det,
    Ptr<BOWImgDescriptorExtractor> bowExt,
    vector<Mat>& hists)
{
    // TODO: Use bowExt to get histograms for each image.
    return true;
}

// 7. Build database matrix
bool assembleDatabase(const vector<Mat>& hists,
    Mat& db)
{
    // TODO: vconcat all hist mats into db.
    return true;
}

// 8. Match query
bool matchQuery(const Mat& query,
    const Mat& db,
    Ptr<DescriptorMatcher> matcher,
    DMatch& best)
{
    // TODO: Find best match between query and db rows.
    return true;
}

int main()
{
    char fname[MAX_PATH];
    cout << "Select CIFAR-10 .bin file...\n";
    if (!openFileDlg(fname)) return -1;

    // Step 1: Load CIFAR-10 batch
    vector<Mat> colorImgs; vector<int> labels;
    if (!loadCIFAR10Batch(fname, colorImgs, labels)) return -1;

    // Show the first loaded color image
    imshow("1 - Original Color Image", colorImgs[0]);
    waitKey(0);

    // Step 2: Convert to grayscale
    vector<Mat> grayImgs;
    if (!convertToGrayscale(colorImgs, grayImgs)) return -1;

    // Show the first grayscale image
    imshow("2 - Grayscale Image", grayImgs[0]);
    waitKey(0);

    // Step 3: Extract SIFT features
    auto detector = SIFT::create();
    vector<Mat> descs;
    if (!detectAndCompute(grayImgs, detector, descs)) return -1;

    // Show keypoints on the first image
    vector<KeyPoint> kpts;
    detector->detect(grayImgs[0], kpts);
    Mat vis;
    drawKeypoints(colorImgs[0], kpts, vis);
    imshow("3 - Keypoints on Color Image", vis);
    waitKey(0);

    // Placeholder messages for the rest
    cout << "\n--- Remaining Steps (Coming Next) ---\n";
    cout << "4. Build visual vocabulary (k-means clustering)...\n";
    cout << "5. Initialize Bag-of-Words extractor...\n";
    cout << "6. Compute BoW histograms for all images...\n";
    cout << "7. Build a database of all BoW vectors...\n";
    cout << "8. Match query image against the database...\n";
    cout << "--------------------------------------\n";

    return 0;
}
