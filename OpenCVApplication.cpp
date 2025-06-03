// 3-Scene Bag-of-Words Clustering (tuned for class-aligned clusters)
// - NUM_CLUSTERS = 3 to match three folders
// - VOCAB_SIZE = 200 for a richer vocabulary
// - MAX_IMAGES_PER_CLASS = 100 for faster iteration
// - After clustering, save 5 representative images per cluster (closest to cluster center)

#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/features2d.hpp>   // SIFT in OpenCV 4.x
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <direct.h>
#include <errno.h>
#include <chrono>
#include <windows.h>
#include <io.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

// ------------------------------
// Configuration (numeric folders)
// ------------------------------
const int NUM_CLUSTERS = 3;               // One cluster per folder
const int MAX_IMAGES_PER_CLASS = 100;     // Limit per numeric folder for speed
const int MIN_IMAGES_PER_CLASS = 20;      // Minimum images needed per folder
const int VOCAB_SIZE = 200;               // Larger vocabulary
const int MAX_FEATURES_PER_IMAGE = 100;   // SIFT keypoints per image

// Only process these three numeric folders:
const set<string> SELECTED_CLASSES = { "00", "01", "02" };

// ------------------------------
// Utility Functions
// ------------------------------
bool makeDir(const string& dir) {
    if (_mkdir(dir.c_str()) != 0 && errno != EEXIST) {
        cerr << "ERROR: cannot create directory " << dir << " (" << errno << ")\n";
        return false;
    }
    return true;
}

bool directoryExists(const string& path) {
    DWORD attribs = GetFileAttributesA(path.c_str());
    return (attribs != INVALID_FILE_ATTRIBUTES && (attribs & FILE_ATTRIBUTE_DIRECTORY));
}

vector<string> getSubdirectories(const string& path) {
    vector<string> subdirs;
    string searchPath = path + "\\*";

    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if ((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                strcmp(findData.cFileName, ".") != 0 &&
                strcmp(findData.cFileName, "..") != 0) {
                subdirs.push_back(string(findData.cFileName));
            }
        } while (FindNextFileA(hFind, &findData));
        FindClose(hFind);
    }
    return subdirs;
}

vector<string> getImageFiles(const string& path) {
    vector<string> imageFiles;
    string searchPath = path + "\\*.*";

    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                string filename = string(findData.cFileName);
                string extension = filename.substr(filename.find_last_of(".") + 1);
                transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                if (extension == "jpg" || extension == "jpeg" ||
                    extension == "png" || extension == "bmp") {
                    imageFiles.push_back(path + "\\" + filename);
                }
            }
        } while (FindNextFileA(hFind, &findData));
        FindClose(hFind);
    }
    return imageFiles;
}

// ------------------------------
// Step 1: Load 3 Numeric Folders
// ------------------------------
bool load3SceneNumeric(const string& datasetPath,
    vector<Mat>& images,
    vector<int>& labels,
    vector<string>& classNames)
{
    cout << "Loading selected folders from: " << datasetPath << "\n";

    images.clear();
    labels.clear();
    classNames.clear();

    if (!directoryExists(datasetPath)) {
        cerr << "Dataset path does not exist: " << datasetPath << "\n";
        return false;
    }

    vector<string> classDirs = getSubdirectories(datasetPath);
    if (classDirs.empty()) {
        cerr << "No subdirectories found in: " << datasetPath << "\n";
        return false;
    }

    // Debug: print all found numeric folders
    cout << "  -> Found folders:\n";
    for (auto& d : classDirs) {
        cout << "       " << d << "\n";
    }

    int classId = 0;
    int totalImages = 0;

    for (const string& rawName : classDirs) {
        // Only process if folder name is in SELECTED_CLASSES
        if (SELECTED_CLASSES.count(rawName) == 0) {
            continue;
        }

        cout << "Processing folder: " << rawName << " (ID: " << classId << ")\n";
        string classPath = datasetPath + "\\" + rawName;
        vector<string> imageFiles = getImageFiles(classPath);

        cout << "  Found " << imageFiles.size()
            << " image files in folder " << rawName << "\n";

        if (imageFiles.size() < MIN_IMAGES_PER_CLASS) {
            cout << "  Skipping folder " << rawName
                << " (only " << imageFiles.size() << " images)\n";
            continue;
        }

        int loadedFromClass = 0;
        for (size_t i = 0;
            i < imageFiles.size() && loadedFromClass < MAX_IMAGES_PER_CLASS;
            ++i)
        {
            Mat img = imread(imageFiles[i], IMREAD_COLOR);
            if (img.empty()) {
                cout << "    Failed to load: " << imageFiles[i] << "\n";
                continue;
            }

            // Resize to 256×256 while preserving aspect ratio
            int targetSize = 256;
            double scale = (double)targetSize / max(img.rows, img.cols);
            int newW = (int)(img.cols * scale);
            int newH = (int)(img.rows * scale);

            Mat resized;
            resize(img, resized, Size(newW, newH), 0, 0, INTER_LINEAR);

            // Pad to 256×256 if needed
            Mat padded = Mat::zeros(targetSize, targetSize, CV_8UC3);
            int offsetX = (targetSize - newW) / 2;
            int offsetY = (targetSize - newH) / 2;
            resized.copyTo(padded(Rect(offsetX, offsetY, newW, newH)));

            images.push_back(padded);
            labels.push_back(classId);
            loadedFromClass++;
            totalImages++;
        }

        cout << "  Loaded " << loadedFromClass
            << " images from folder " << rawName << "\n";

        classNames.push_back(rawName);
        classId++;

        if (classId >= (int)SELECTED_CLASSES.size()) {
            break;
        }
    }

    cout << "Total loaded: " << totalImages
        << " images from " << classId << " folders\n";

    return !images.empty();
}

// -----------------------------------
// Step 2: Convert All to Grayscale
// -----------------------------------
void convertToGrayscale(const vector<Mat>& colorImgs, vector<Mat>& grayImgs) {
    grayImgs.clear();
    grayImgs.resize(colorImgs.size());

    cout << "Converting to grayscale...\n";
    for (size_t i = 0; i < colorImgs.size(); i++) {
        cvtColor(colorImgs[i], grayImgs[i], COLOR_BGR2GRAY);
    }
    cout << "Converted " << grayImgs.size() << " images to grayscale\n";
}

// -----------------------------------------------------
// Step 3: Extract SIFT Features (float descriptors)
// -----------------------------------------------------
void extractSIFTFeatures(const vector<Mat>& grayImgs,
    vector<Mat>& allDescriptors,
    Ptr<Feature2D>& detector)
{
    cout << "Extracting SIFT features...\n";

    detector = SIFT::create(MAX_FEATURES_PER_IMAGE);

    allDescriptors.clear();
    int totalDescriptors = 0;
    int successfulImages = 0;

    for (size_t i = 0; i < grayImgs.size(); ++i) {
        if (i % 50 == 0) {
            cout << "  Processing image " << i << "/" << grayImgs.size() << "\n";
        }

        Mat img = grayImgs[i];

        // Preprocess: blur + CLAHE
        Mat processed;
        GaussianBlur(img, processed, Size(3, 3), 0.5);
        Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
        clahe->apply(processed, processed);

        vector<KeyPoint> keypoints;
        Mat desc;

        detector->detectAndCompute(processed, noArray(), keypoints, desc);

        if (!desc.empty() && desc.rows > 0) {
            allDescriptors.push_back(desc);
            totalDescriptors += desc.rows;
            successfulImages++;
        }
        else {
            // Fallback: grid of keypoints, still compute SIFT
            keypoints.clear();
            for (int y = 20; y < img.rows - 20; y += 40) {
                for (int x = 20; x < img.cols - 20; x += 40) {
                    keypoints.emplace_back(Point2f(x, y), 16.0f);
                }
            }
            if (!keypoints.empty()) {
                detector->compute(processed, keypoints, desc);
                if (!desc.empty()) {
                    allDescriptors.push_back(desc);
                    totalDescriptors += desc.rows;
                    successfulImages++;
                }
            }
        }
    }

    cout << "Extracted " << totalDescriptors << " descriptors from "
        << successfulImages << "/" << grayImgs.size() << " images\n";
}

// --------------------------------------------------
// Step 4: Build Visual Vocabulary (BoW) via k-means
// --------------------------------------------------
Mat buildVocabulary(const vector<Mat>& allDescriptors, int vocabSize = VOCAB_SIZE) {
    cout << "Building visual vocabulary with " << vocabSize << " words...\n";

    if (allDescriptors.empty()) {
        cerr << "No descriptors for vocabulary building\n";
        return Mat();
    }

    // Stack a sample of descriptors into one Mat
    Mat trainingData;
    RNG rng(12345);
    int maxSamples = 30000; // cap total samples

    for (const auto& desc : allDescriptors) {
        if (desc.empty()) continue;
        int sampleCount = min(desc.rows, 20);
        for (int i = 0; i < sampleCount; ++i) {
            if (trainingData.rows >= maxSamples) break;
            int idx = rng.uniform(0, desc.rows);
            trainingData.push_back(desc.row(idx));
        }
        if (trainingData.rows >= maxSamples) break;
    }

    if (trainingData.rows < vocabSize) {
        int newSize = max(10, trainingData.rows / 2);
        cout << "  Reducing vocabulary size to " << newSize << "\n";
        vocabSize = newSize;
    }

    // Ensure trainingData is float (SIFT outputs CV_32F)
    Mat trainingDataFloat;
    trainingData.convertTo(trainingDataFloat, CV_32F);

    TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 80, 0.001);
    BOWKMeansTrainer bowTrainer(vocabSize, criteria, 3, KMEANS_PP_CENTERS);
    Mat vocabulary = bowTrainer.cluster(trainingDataFloat);

    cout << "Built vocabulary with " << vocabulary.rows << " words\n";
    return vocabulary;
}

// ----------------------------------------------------------
// Step 5: Compute BoW Histograms for Each Image
// ----------------------------------------------------------
void computeBoWHistograms(const vector<Mat>& grayImgs,
    const Mat& vocabulary,
    Ptr<Feature2D>& detector,
    Mat& bowFeatures)
{
    cout << "Computing BoW histograms...\n";

    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2);
    Ptr<BOWImgDescriptorExtractor> bowExtractor =
        makePtr<BOWImgDescriptorExtractor>(detector, matcher);
    bowExtractor->setVocabulary(vocabulary);

    int vocabSize = vocabulary.rows;
    bowFeatures = Mat::zeros((int)grayImgs.size(), vocabSize, CV_32F);

    for (size_t i = 0; i < grayImgs.size(); ++i) {
        if (i % 50 == 0) {
            cout << "  Computing histogram " << i << "/" << grayImgs.size() << "\n";
        }

        Mat img = grayImgs[i];
        Mat processed;
        GaussianBlur(img, processed, Size(3, 3), 0.5);
        Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
        clahe->apply(processed, processed);

        vector<KeyPoint> keypoints;
        detector->detect(processed, keypoints);

        if (keypoints.size() < 15) {
            for (int y = 20; y < img.rows - 20; y += 40) {
                for (int x = 20; x < img.cols - 20; x += 40) {
                    keypoints.emplace_back(Point2f(x, y), 16.0f);
                }
            }
        }

        Mat bowHist;
        try {
            bowExtractor->compute(processed, keypoints, bowHist);
        }
        catch (...) {
            bowHist = Mat::zeros(1, vocabSize, CV_32F);
        }

        if (bowHist.empty()) {
            bowHist = Mat::ones(1, vocabSize, CV_32F) * (1.0f / vocabSize);
        }
        else {
            normalize(bowHist, bowHist, 1.0, 0.0, NORM_L2);
            double normVal = cv::norm(bowHist);
            if (normVal == 0 || isnan(normVal) || isinf(normVal)) {
                bowHist = Mat::ones(1, vocabSize, CV_32F) * (1.0f / vocabSize);
            }
        }

        bowHist.copyTo(bowFeatures.row((int)i));
    }

    cout << "Computed BoW histograms for " << grayImgs.size() << " images\n";
}

// -------------------------------------------------------
// Step 6: Perform K-means Clustering on BoW Histograms
// -------------------------------------------------------
// Outputs both clusterLabels and clusterCenters.
bool performKMeansClustering(const Mat& bowFeatures,
    int numClusters,
    vector<int>& clusterLabels,
    Mat& clusterCenters)
{
    cout << "Performing K-means clustering with " << numClusters << " clusters...\n";

    if (bowFeatures.empty() || bowFeatures.rows < numClusters) {
        cerr << "Not enough samples for clustering\n";
        return false;
    }

    Mat labels, centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 80, 1.0);

    try {
        double compactness = kmeans(bowFeatures, numClusters, labels,
            criteria, 5, KMEANS_PP_CENTERS, centers);
        cout << "K-means completed with compactness: " << compactness << "\n";
    }
    catch (cv::Exception& e) {
        cerr << "K-means failed: " << e.what() << "\n";
        return false;
    }

    clusterLabels.resize(labels.rows);
    for (int i = 0; i < labels.rows; ++i) {
        clusterLabels[i] = labels.at<int>(i, 0);
    }
    clusterCenters = centers.clone();
    return true;
}

// -------------------------------------------------
// Step 7: Save Clustered Results to Output Folder
// -------------------------------------------------
void saveClusteredResults(const vector<Mat>& images,
    const vector<int>& clusterLabels,
    const vector<int>& originalLabels,
    const vector<string>& classNames)
{
    const string outDir = "3scene_clustered";
    if (!makeDir(outDir)) return;

    for (int i = 0; i < NUM_CLUSTERS; ++i) {
        string clusterDir = outDir + "\\cluster_" + to_string(i);
        makeDir(clusterDir);
    }

    cout << "Saving clustered images...\n";
    for (size_t i = 0; i < images.size(); ++i) {
        if (i % 50 == 0) {
            cout << "  Saving image " << i << "/" << images.size() << "\n";
        }

        int cluster = clusterLabels[i];
        int originalLabel = originalLabels[i];
        string originalClass = (originalLabel < classNames.size())
            ? classNames[originalLabel] : "unknown";

        char buf[256];
        snprintf(buf, sizeof(buf), "img_%04zu_orig_%s.jpg", i, originalClass.c_str());
        string filepath = outDir + "\\cluster_" + to_string(cluster) + "\\" + buf;
        imwrite(filepath, images[i]);
    }

    cout << "Saved " << images.size() << " images to '" << outDir << "'\n";

    // Basic cluster distribution analysis
    cout << "\n=== 3-Scene BoW Clustering Analysis ===\n";
    vector<int> clusterCounts(NUM_CLUSTERS, 0);
    for (int label : clusterLabels) {
        if (label >= 0 && label < NUM_CLUSTERS) {
            clusterCounts[label]++;
        }
    }
    cout << "\nCluster Distribution:\n";
    for (int i = 0; i < NUM_CLUSTERS; ++i) {
        cout << "  Cluster " << i << ": " << clusterCounts[i] << " images\n";
    }

    cout << "\nCluster vs Original Folder Analysis:\n";
    for (int c = 0; c < NUM_CLUSTERS; ++c) {
        cout << "\nCluster " << c << " contains:\n";
        map<string, int> folderCountInCluster;
        for (size_t i = 0; i < clusterLabels.size(); ++i) {
            if (clusterLabels[i] == c) {
                string cls = (originalLabels[i] < classNames.size())
                    ? classNames[originalLabels[i]] : "unknown";
                folderCountInCluster[cls]++;
            }
        }
        vector<pair<int, string>> sortedFolders;
        for (const auto& p : folderCountInCluster) {
            sortedFolders.push_back({ p.second, p.first });
        }
        sort(sortedFolders.rbegin(), sortedFolders.rend());
        int shown = 0;
        for (const auto& p : sortedFolders) {
            cout << "  " << p.second << ": " << p.first << " images\n";
            if (++shown >= 5) break;
        }
    }
}

// ---------------------------------------------------------------
// Step 8: Save 5 Representative Images per Cluster
//    (closest to cluster center in histogram space)
// ---------------------------------------------------------------
void saveClusterRepresentatives(const vector<Mat>& images,
    const Mat& bowFeatures,
    const vector<int>& clusterLabels,
    const Mat& clusterCenters,
    const vector<string>& classNames,
    const vector<int>& originalLabels)
{
    const string outDir = "3scene_clustered";
    // For each cluster, collect indices, compute distances, pick top 5
    for (int c = 0; c < NUM_CLUSTERS; ++c) {
        // Gather indices belonging to cluster c
        vector<pair<double, int>> dists; // (distance, index)
        for (int i = 0; i < (int)clusterLabels.size(); ++i) {
            if (clusterLabels[i] == c) {
                double dist = norm(bowFeatures.row(i), clusterCenters.row(c), NORM_L2);
                dists.push_back({ dist, i });
            }
        }
        if (dists.empty()) continue;
        // Sort by ascending distance
        sort(dists.begin(), dists.end(),
            [](auto& a, auto& b) { return a.first < b.first; });

        // Create "representatives" subfolder
        string repDir = outDir + "\\cluster_" + to_string(c) + "\\representatives";
        makeDir(repDir);

        // Save up to 5 nearest images
        int toSave = min(5, (int)dists.size());
        for (int k = 0; k < toSave; ++k) {
            int idx = dists[k].second;
            int origLabel = originalLabels[idx];
            string origName = (origLabel < classNames.size())
                ? classNames[origLabel] : "unknown";
            char buf[256];
            snprintf(buf, sizeof(buf),
                "rep_%02d_idx_%04d_from_%s.jpg",
                k, idx, origName.c_str());
            string filepath = repDir + "\\" + string(buf);
            imwrite(filepath, images[idx]);
        }
        cout << "  Saved " << toSave << " representatives for cluster " << c << "\n";
    }
}

// -------------------------------------------------------------
// Main Pipeline: Orchestrates all Steps for 3 Numeric Folders
// -------------------------------------------------------------
bool cluster3SceneNumeric(const string& datasetPath) {
    auto startTime = chrono::high_resolution_clock::now();

    // ===== Step 1: Load 3 Numeric Folders =====
    vector<Mat> images;
    vector<int> labels;
    vector<string> classNames;
    if (!load3SceneNumeric(datasetPath, images, labels, classNames)) {
        cerr << "Failed to load selected folders\n";
        return false;
    }

    // ===== Step 2: Convert to Grayscale =====
    vector<Mat> grayImages;
    convertToGrayscale(images, grayImages);

    // ===== Step 3: Extract SIFT Features =====
    vector<Mat> allDescriptors;
    Ptr<Feature2D> detector;
    extractSIFTFeatures(grayImages, allDescriptors, detector);
    if (allDescriptors.empty()) {
        cerr << "Failed to extract features\n";
        return false;
    }

    // ===== Step 4: Build Vocabulary =====
    Mat vocabulary = buildVocabulary(allDescriptors, VOCAB_SIZE);
    if (vocabulary.empty()) {
        cerr << "Failed to build vocabulary\n";
        return false;
    }

    // ===== Step 5: Compute BoW Histograms =====
    Mat bowFeatures;
    computeBoWHistograms(grayImages, vocabulary, detector, bowFeatures);

    // ===== Step 6: Perform K-means Clustering =====
    vector<int> clusterLabels;
    Mat clusterCenters;
    if (!performKMeansClustering(bowFeatures, NUM_CLUSTERS, clusterLabels, clusterCenters)) {
        cerr << "Clustering failed\n";
        return false;
    }

    // ===== Step 7: Save Clustered Results =====
    saveClusteredResults(images, clusterLabels, labels, classNames);

    // ===== Step 8: Save 5 Representative Images per Cluster =====
    saveClusterRepresentatives(images, bowFeatures, clusterLabels,
        clusterCenters, classNames, labels);

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
    cout << "\n3-Scene BoW clustering completed in " << duration << " seconds\n";

    return true;
}

int main() {
    cout << "=== 3-Scene Bag-of-Words Clustering ===\n";
    cout << "This will cluster images from three numeric folders (\"00\", \"01\", \"02\")\n\n";

    // Get dataset path from user
    string datasetPath;
    cout << "Enter the path to your 15-Scene dataset directory\n";
    cout << "(Should contain numeric subfolders like '00', '01', '02', etc.):\n";
    getline(cin, datasetPath);

    if (datasetPath.empty()) {
        cout << "No path provided. Exiting.\n";
        return -1;
    }

    if (!cluster3SceneNumeric(datasetPath)) {
        cout << "Error during clustering\n";
        return -1;
    }

    cout << "\nClustering completed successfully!\n";
    cout << "Check '3scene_clustered' directory for results\n";
    cout << "  - Each cluster’s folder contains all images assigned to it\n";
    cout << "  - Under each cluster, 'representatives' holds 5 images closest to the cluster center\n";

    return 0;
}
