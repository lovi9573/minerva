/*
 * unittest_fixedpoint.cpp
 *
 *  Created on: Jul 29, 2015
 *      Author: jlovitt
 */


#define FIXED_POINT_FRACTION_WIDTH_N 14
#define FIXED_POINT_WORD_LENGTH_N 15
#define FIXED_POINT_TYPE int16_t
#define FIXED_POINT_DOUBLE_WIDE_TYPE int32_t

#define FIXED_POINT

#include <stdint.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <common/fixedpoint.h>
#include "unittest_main.h"
#include <time.h>
#include <sys/time.h>

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

#define FRACTION_WIDTH 12
#define SINGLETYPE int16_t
#define DOUBLETYPE int32_t

typedef FixedPoint<DOUBLETYPE,SINGLETYPE,15,0> FP;
typedef FixedPoint<DOUBLETYPE,SINGLETYPE,FIXED_POINT_WORD_LENGTH_N,FRACTION_WIDTH> FP8;

#define MAXM(X, Y) (((X) > (Y)) ? (X) : (Y))

inline float addfloat(float a, float b, float c){
	return a * b + c;
}

inline FP8 addfixed(FP8 a, FP8 b, FP8 c){
	return a * b + c;
}

TEST(FixedPoint, MulAddSpeed){
	int number_random_tests = 1000000;
	double a = get_wall_time();	
	for(int i = 0; i < number_random_tests; i++){
		addfloat(static_cast<float>(rand())/static_cast<float>(RAND_MAX),
				static_cast<float>(rand())/static_cast<float>(RAND_MAX),
				static_cast<float>(rand())/static_cast<float>(RAND_MAX));
	}
	a = get_wall_time() - a;
	std::cout << "float time: " << a << " \n";
	
	double b = get_wall_time();	
	for(int i = 0; i < number_random_tests; i++){
		addfixed(static_cast<float>(rand())/static_cast<float>(RAND_MAX),
				static_cast<float>(rand())/static_cast<float>(RAND_MAX),
				static_cast<float>(rand())/static_cast<float>(RAND_MAX));
	}
	b = get_wall_time() - b;
	std::cout << "fixed time: " << b << " \n";
	EXPECT_LT((b/a), 3.0);
	std::cout << "factor: " << (b/a) << "\n";
}



