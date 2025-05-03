#include "stdafx.h"
#include "common.h"
#include "OpenCVApplication.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <direct.h>
#include <errno.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

// 0) CIFAR-10 class names for object-based sorting
const vector<string> LABEL_NAMES = {
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
};

// Utility: create directory if it doesn't exist
bool makeDir(const string& dir) {
    if (_mkdir(dir.c_str()) != 0 && errno != EEXIST) {
        cerr << "ERROR: cannot create directory " << dir << " (" << errno << ")\n";
        return false;
    }
    return true;
}

// STEP 1: Load CIFAR-10 batch file into color images and labels
bool loadCIFAR10Batch(const string& filename,
    vector<Mat>& images,
    vector<int>& labels)
{
    const int SZ = 3072;          // bytes per image (3 x 32 x 32)
    const int H = 32, W = 32;
    const int N = 10000;          // images per batch

    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Can't open " << filename << "\n";
        return false;
    }

    images.clear(); labels.clear();
    images.reserve(N); labels.reserve(N);
    vector<uint8_t> buf(SZ);

    for (int i = 0; i < N; ++i) {
        // read label byte
        uint8_t l;
        file.read(reinterpret_cast<char*>(&l), 1);
        labels.push_back(static_cast<int>(l));

        // read image bytes
        file.read(reinterpret_cast<char*>(buf.data()), SZ);

        // convert to BGR Mat
        Mat img(H, W, CV_8UC3);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                int idx = r * W + c;
                img.at<Vec3b>(r, c) = Vec3b(
                    buf[2 * W * H + idx],  // blue
                    buf[W * H + idx],      // green
                    buf[idx]               // red
                );
            }
        }
        images.push_back(img);
    }

    cout << "Loaded " << images.size() << " images and labels from " << filename << "\n";
    return true;
}

int main()
{
    // File selection dialog for CIFAR-10 binary batch
    char fname[MAX_PATH];
    cout << "Select CIFAR-10 .bin file...\n";
    if (!openFileDlg(fname)) return -1;

    // ------------ Pipeline Variables ------------
    vector<Mat> colorImgs;       // original color images
    vector<int> labels;          // true class labels
    vector<Mat> grayImgs;        // grayscale images
    vector<Mat> descs;           // SIFT descriptors per image
    Mat vocab;                   // visual vocabulary (words x descriptor dim)
    Ptr<Feature2D> detector;     // SIFT feature detector
    Ptr<DescriptorMatcher> matcher;
    Ptr<BOWImgDescriptorExtractor> bowExtractor;
    vector<Mat> bOWHists;        // BoW histograms per image
    Mat db;                      // database matrix of all histograms

    // 1) LOAD DATA
    if (!loadCIFAR10Batch(fname, colorImgs, labels)) return -1;

    // 2) SORT BY OBJECT CLASS (ground truth labels)
    cout << "Grouping images by true object class...\n";
    string objDir = "object_sorted";
    if (!makeDir(objDir)) return -1;
    for (size_t i = 0; i < colorImgs.size(); ++i) {
        int lbl = labels[i];
        string classDir = objDir + "/" + LABEL_NAMES[lbl];
        if (!makeDir(classDir)) return -1;
        char buf[64];
        snprintf(buf, sizeof(buf), "img_%04zu.png", i);
        imwrite(classDir + "/" + buf, colorImgs[i]);
    }
    cout << "Saved class-grouped images under '" << objDir << "'\n";

    // 3) CONVERT TO GRAYSCALE (prepare for feature detection)
    cout << "Converting to grayscale...\n";
    if (colorImgs.empty()) { cerr << "No images to process\n"; return -1; }
    grayImgs.reserve(colorImgs.size());
    for (auto& img : colorImgs) {
        Mat g;
        cvtColor(img, g, COLOR_BGR2GRAY);
        grayImgs.push_back(g);
    }
    cout << "Converted " << grayImgs.size() << " images to grayscale\n";

    // 4) EXTRACT SIFT FEATURES
    cout << "Detecting and computing SIFT descriptors...\n";
    detector = SIFT::create();
    for (auto& g : grayImgs) {
        vector<KeyPoint> keypoints;
        Mat descriptors;
        detector->detectAndCompute(g, noArray(), keypoints, descriptors);
        if (!descriptors.empty()) descs.push_back(descriptors);
    }
    cout << "Extracted descriptors from " << descs.size() << " images\n";

    // 5) BUILD VISUAL VOCABULARY via k-means
    cout << "Building visual vocabulary (k-means clustering)...\n";
    int dictSize = 50;
    {
        // stack all descriptors vertically
        Mat allDesc;
        for (auto& d : descs) allDesc.push_back(d);
        TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.001);
        BOWKMeansTrainer bowTrainer(dictSize, tc, 1, KMEANS_PP_CENTERS);
        vocab = bowTrainer.cluster(allDesc);
    }
    cout << "Vocabulary of " << vocab.rows << " words created\n";

    // 6) SETUP BoW EXTRACTOR
    cout << "Initializing Bag-of-Words extractor...\n";
    matcher = BFMatcher::create(NORM_L2);
    bowExtractor = makePtr<BOWImgDescriptorExtractor>(detector, matcher);
    bowExtractor->setVocabulary(vocab);
    cout << "BoW extractor ready\n";

    // 7) COMPUTE BoW HISTOGRAMS
    cout << "Computing BoW histograms...\n";
    for (auto& g : grayImgs) {
        vector<KeyPoint> keypoints;
        detector->detect(g, keypoints);
        Mat hist;
        bowExtractor->compute(g, keypoints, hist);
        if (!hist.empty()) bOWHists.push_back(hist);
    }
    cout << "Computed histograms for " << bOWHists.size() << " images\n";

    // 8) ASSEMBLE DATABASE MATRIX
    cout << "Assembling database matrix...\n";
    if (!bOWHists.empty()) {
        db = bOWHists[0].reshape(1, 1);
        for (size_t i = 1; i < bOWHists.size(); ++i) {
            Mat row = bOWHists[i].reshape(1, 1);
            vconcat(db, row, db);
        }
    }
    cout << "Database assembled with " << db.rows << " entries\n";

    // 9) SORT IMAGES BY VISUAL-WORD DOMINANCE & STRENGTH
    cout << "Sorting images by visual word and strength...\n";
    struct ImgInfo { int word, idx; float weight; };
    vector<ImgInfo> info;
    for (int i = 0; i < db.rows; ++i) {
        Point maxLoc;
        double maxVal;
        minMaxLoc(bOWHists[i], nullptr, &maxVal, nullptr, &maxLoc);
        info.push_back({ maxLoc.x, i, static_cast<float>(maxVal) });
    }
    sort(info.begin(), info.end(), [](auto& a, auto& b) {
        if (a.word != b.word) return a.word < b.word;
        return a.weight > b.weight;
        });

    string outDir = "sorted_images";
    if (!makeDir(outDir)) return -1;
    for (auto& im : info) {
        string wordDir = outDir + "/word_" + to_string(im.word);
        if (!makeDir(wordDir)) return -1;
        char buf[64];
        snprintf(buf, sizeof(buf), "img_%04d.png", im.idx);
        imwrite(wordDir + "/" + buf, colorImgs[im.idx]);
    }
    cout << "Saved images sorted by visual word under '" << outDir << "'\n";

    return 0;
}
