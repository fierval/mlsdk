
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include <opencv2\opencv.hpp>
#include <thrust\device_vector.h>
#include <thrust\copy.h>
#include <ctime>

__global__ void transformKernel(unsigned char * mask, unsigned char * label, int cols, int size, int nClasses)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
    {

        int r = i / cols;
        int c = i % cols;
        unsigned char plane = mask[i];
        unsigned char channel = plane >= nClasses ? 0 : plane;
        int idx = channel + c * nClasses + r * cols * nClasses;
        label[idx] = 1;
    }
}

using namespace std;
using namespace cv;

const String keys = {
    "{help h usage ? |      | print this message   }"
    "{@image1        |      | image1 for compare   }"
    "{@classes       |   2  | number of classes   }"
};

int convertMask(Mat& img, int nClasses, bool debug, bool onGpu)
{
    vector<uchar> imgArray;
    vector<vector<vector<uchar>>> labels3d(img.rows);

    // Convert to a 3D representation
    for (int i = 0; i < img.rows; i++)
    {
        labels3d[i].resize(img.cols);
        for (int j = 0; j < img.cols; j++)
        {
            labels3d[i][j].resize(nClasses);
        }
    }

    if (debug)
    {
        cout << "Classses: " << nClasses << endl;
        cout << "Size: " << img.rows * img.cols << endl;
        cout << "Rows: " << img.rows << endl << "Cols: " << img.cols << endl;
    }


    if (onGpu)
    {

        // copy from Mat -> vector
        if (img.isContinuous())
        {
            imgArray.assign(img.datastart, img.dataend);
        }
        else
        {
            for (int i = 0; i < img.rows; i++)
            {
                imgArray.insert(imgArray.end(), img.ptr<uchar>(i), img.ptr<uchar>(i) + img.cols);
            }
        }

        // Use Thrust to allocate device memory because it's easier
        thrust::device_vector<uchar> d_mask(imgArray);
        thrust::device_vector<uchar> d_label(d_mask.size() * nClasses, 0U);

        int blockSize = 256;
        int gridSize = (d_mask.size() + blockSize - 1) / blockSize;

        if (debug)
        {
            cout << "Grid: " << gridSize << endl << "Block: " << blockSize << endl;
        }

        uchar * ptrMask = thrust::raw_pointer_cast(d_mask.data());
        uchar * ptrLabels = thrust::raw_pointer_cast(d_label.data());

        transformKernel << <gridSize, blockSize >> > (ptrMask, ptrLabels, img.cols, img.rows * img.cols, nClasses);

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return 1;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel\n", cudaStatus);
            return 1;
        }

        //Need to copy results back
        thrust::host_vector<uchar> labels(d_label);

        if (debug)
        {
            vector<uchar> bar;
            auto it = std::copy_if(labels.begin(), labels.end(), std::back_inserter(bar), [](int i) {return i > 0; });
            if (bar.size() != labels.size() / 2)
            {
                cout << "Kernel error";
            }

            cout << "Kernel completed successfully" << endl;
        }

        for (int i = 0; i < img.rows * img.cols * nClasses; i++)
        {
            int classs = i % nClasses;
            int col = (i / nClasses) % img.cols;
            int row = i / nClasses / img.cols;
            labels3d[row][col][classs] = labels[i];
        }

    }
    else
    {
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                int classs = img.at<uchar>(i,j) < nClasses ? img.at<uchar>(i, j) : 0;
                labels3d[i][j][classs] = 1;
            }

        }

    }

    return 0;
}
int main(int argc, char* argv[])
{

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String imgFile = parser.get<String>(0);
    int nClasses = parser.get<int>(1);

    Mat img = imread(imgFile, IMREAD_GRAYSCALE);

    int sizes[] = { 64, 128, 256, 512, 1024, 2048, 4096 };

    auto l = convertMask(img, nClasses, true, true);

    for (auto s : sizes)
    {
        Mat im;
        resize(img, im, Size(s, s));

        // GPU

        double duration = 0;
        int n = 5;
        for (int i = 0; i < n; i++)
        {
            int start = clock();
            convertMask(im, nClasses, false, true);
            int stop = clock();
            duration += (stop - start) / double(CLOCKS_PER_SEC) * 1000;
        }

        double durationCpu = 0;
        // CPU
        for (int i = 0; i < n; i++)
        {
            int start = clock();

            convertMask(im, nClasses, false, false);

            int stop = clock();

            durationCpu += (stop - start) / double(CLOCKS_PER_SEC) * 1000;
        }
        cout << "Size: " << s << endl << "\tGPU: " << duration / n << endl << "\tCPU " << durationCpu / n << endl;
    }

}
