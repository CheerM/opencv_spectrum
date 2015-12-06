#include <stdio.h>
#include <iostream>
#include <complex>
#include <math.h>
#include <opencv2/opencv.hpp>

#define PI 3.1415926

using namespace cv;
using namespace std;

void release(complex<double>** input, int x) {
	for (int i = 0; i < x; i ++)
		delete input[i];
	delete input;
}

complex<double>** dft(complex<double>** origin, bool flags, int row, int col) {
	//y方向DFT / IDFT
	complex<double>** output1_complex = new complex<double>*[row];
	for (int x = 0; x < row; x ++) {
		output1_complex[x] = new complex<double>[col];
		for (int v = 0; v < col; v ++) {
			complex<double> temp_y(0, 0);
			for (int y = 0; y < col; y ++) {
				complex<double> tt;
				if (flags)
					tt = complex<double>(cos(2*PI*v*y / col), (-1) * sin(2*PI*v*y / col));
				else
					tt = complex<double>(cos(2*PI*v*y / col), sin(2*PI*v*y / col));
				temp_y += origin[x][y] * tt;
			}
			output1_complex[x][v] = temp_y;
		}
	}
	release(origin, row);

	//X方向DFT / IDFT
	complex<double>** output2_complex = new complex<double>*[row];
	for (int u = 0; u < row; u ++) {
		output2_complex[u] = new complex<double>[col];
		for (int v = 0; v < col; v ++) {
			complex<double> temp_x(0, 0);
			for (int x = 0; x < row; x ++) {
				complex<double> tt;
				if (flags)
					tt = complex<double>(cos(2*PI*u*x / row), (-1) * sin(2*PI*u*x / row));
				else 
					tt = complex<double>(cos(2*PI*u*x / row), sin(2*PI*u*x / row));
				temp_x += output1_complex[x][v] * tt;
			}
			output2_complex[u][v] = temp_x;
		}
	}
	release(output1_complex, row);
	return output2_complex;
}

//把DFT变换后的数组取频谱值、对数化、标定，最后转换成Mat形式
Mat show_dft(complex<double>** dft, int row, int col) {
	Mat output = Mat_<uchar>(row, col);
	double max = 0;
	double output_double[row][col];
	for (int i = 0; i < row; i ++) {
		for (int j = 0; j < col; j ++) {
			output_double[i][j] = log(sqrt(pow(dft[i][j].real(), 2) + pow(dft[i][j].imag(), 2)) + 1);
			if (max < output_double[i][j]) max = output_double[i][j];
		}
	}
	//release(dft, row);

	//标定
	double rate = 255/max;
	for (int i = 0; i < row; i ++) {
		uchar* data = output.ptr<uchar>(i);
		for (int j = 0; j < col; j ++) {
			data[j] = (uchar)(output_double[i][j]*rate);
		}
	}
	return output;
}

//把IDFT变换后的数组转换成Mat形式，identify==true表示需要去中心化, demarcate标定
Mat show_idft(complex<double>** idft, int row, int col, bool identify, bool demarcate) {
	Mat output = Mat_<uchar>(row, col);
	double max = 0;
	for (int i = 0; i < row; i ++) {
		for (int j = 0; j < col; j ++) {
			if (max < idft[i][j].real()) max = idft[i][j].real();
		}
	}
	cout << "max idft = " << max << endl;

	double rate = 255 / max;
	double temp;
	for (int i = 0; i < row; i ++) {
		uchar* data = output.ptr<uchar>(i);
		for (int j = 0; j < col; j ++) {
			if (identify == false || (i + j) % 2 == 0)
				temp = idft[i][j].real();
			else if (identify == true && (i + j) % 2 == 1)
				temp = (-1) * idft[i][j].real();
			if (demarcate) data[j] = (uchar) (fabs(temp) * rate);
			else data[j] = (uchar) (temp / (row * col));
			if (data[j] < 0) cout << "ERROR!!!!" << endl;
		}
	}
	release(idft, row);
	return output;
}

Mat dft2d(Mat &input, bool flags) {
	int row = input.rows;
	int col = input.cols;

	//中心化乘以（-1）^(x+y)
	complex<double>** output_complex = new complex<double>*[row];
	for (int i = 0; i < row; i ++) {
		output_complex[i] = new complex<double>[col];
		uchar* data = input.ptr<uchar>(i);
		for (int j = 0; j < col; j ++) {
			if ((i + j) % 2 == 0)
				output_complex[i][j] = complex<double>((double)data[j], 0);
			else
				output_complex[i][j] = complex<double>((-1) * (double)(data[j]), 0);
		}
	}

	//DFT
	complex<double>** output2_complex = dft(output_complex, true, row, col);

	Mat output;
	if (flags) {
		output = show_dft(output2_complex, row, col);
	}
	else {
		//IDFT
		complex<double>** output5_complex = dft(output2_complex, false, row, col);
		output = show_idft(output5_complex, row, col, true, false);
	}
	return output;
}

Mat filter2d_freq(Mat &input, double** filter, int size) {
	int row = input.rows;
	int col = input.cols;
	int Row = row;
	int Col = col;

	////零填充
	//把inputImage转成矩阵操作
	complex<double>** input_complex = new complex<double>*[Row];
	for (int i = 0; i < Row; i ++) {
		input_complex[i] = new complex<double>[Col];
		for (int j = 0; j < Col; j ++) {
			if (i < row && j < col){
				if ((i + j) % 2 == 1)
					input_complex[i][j] = complex<double>(-(double)input.ptr<uchar>(i)[j], 0);
				else 
					input_complex[i][j] = complex<double>((double)input.ptr<uchar>(i)[j], 0);
			}
			else 
				input_complex[i][j] = complex<double>(0, 0);
		}
	}

	//把filter转成矩阵操作
	complex<double>** filter_complex = new complex<double>*[Row];
	for (int i = 0; i < Row; i ++) {
		filter_complex[i] = new complex<double>[Col];
		for (int j = 0; j < Col; j ++) {
			if (i < size && j < size) {
				if ((i + j) % 2 == 1)
					filter_complex[i][j] = complex<double>(-filter[i][j], 0);
				else 
					filter_complex[i][j] = complex<double>(filter[i][j], 0);
			}
			else
				filter_complex[i][j] = complex<double>(0, 0); 
		}
	}

	//filter DFT
	complex<double>** filter_dft = dft(filter_complex, true, Row, Col);
	//Mat DFT_filter = show_dft(filter_dft, Row, Col);
	//imshow("Display Filter Image", DFT_filter);
	
	//input image DFT
	complex<double>** input_dft = dft(input_complex, true, Row, Col);
	//Mat DFT_input = show_dft(input_dft, row, col);
	//imshow("Display DIF input Image", DFT_input);

	//点乘
	complex<double>** idft_complex = new complex<double>*[Row];
	for (int i = 0; i < Row; i ++) {
		idft_complex[i] = new complex<double>[Col];
		for (int j = 0; j < Col; j ++) {
			idft_complex[i][j] = input_dft[i][j] * filter_dft[i][j];
		}
	}
	//Mat DFT_= show_dft(idft_complex, row, col);
	//imshow("Display DIF Image", DFT_);
	release(input_dft, Row);
	release(filter_dft, Row);

	complex<double>** idft3_complex = dft(idft_complex, false, Row, Col);
	Mat output = show_idft(idft3_complex, Row, Col, true, true);
	return output;
}

int main(int argc, char** argv )
{
	//读入07.png
	Mat image;
	image = imread( "07.png", -1 );
	if ( !image.data )
	{
		printf("No image data \n");
		return -1;
	}
	namedWindow("Display input Image", WINDOW_AUTOSIZE );
	imshow("Display input Image", image);

	//DFT2d
	Mat dft_image = dft2d(image, true);
	namedWindow("Display DFT Image", WINDOW_AUTOSIZE );
	imshow("Display DFT Image", dft_image);

	// IDFT2d
	Mat idft_image = dft2d(image, false);
	namedWindow("Display IDFT Image", WINDOW_AUTOSIZE );
	imshow("Display IDFT Image", idft_image);

	//average filter
	double** filter1 = new double*[7];
	for (int i = 0; i < 7; i ++) {
		filter1[i] = new double[7];
		for (int j = 0; j < 7; j ++) {
			filter1[i][j] = 1;
		}
	}
	Mat average_image = filter2d_freq(image, filter1, 7);
	namedWindow("Display average filter Image", WINDOW_AUTOSIZE );
	imshow("Display average filter Image", average_image);

	//3*3 Laplacian filter
	double** filter2 = new double*[3];
	for (int i = 0; i < 3; i ++) {
		filter2[i] = new double[3];
		for (int j = 0; j < 3; j ++) {
			if (i == 1 && j == 1) filter2[i][j] = 8;
			else filter2[i][j] = -1;
		}
	}
	Mat Laplacian_image = filter2d_freq(image, filter2, 3);
	namedWindow("Display Laplacian filter Image", WINDOW_AUTOSIZE );
	imshow("Display Laplacian filter Image", Laplacian_image);

    	waitKey(0);
  	return 0;
}