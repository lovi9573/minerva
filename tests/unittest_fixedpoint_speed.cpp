/*
 * unittest_fixedpoint.cpp
 *
 *  Created on: Jul 29, 2015
 *      Author: jlovitt
 */




#define FIXED_POINT
#include <stdint.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <fixedpoint.h>

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

#define FRACTION_WIDTH 8
#define SINGLETYPE int16_t
#define DOUBLETYPE int32_t

typedef FixedPoint<DOUBLETYPE,SINGLETYPE,0> FP;
typedef FixedPoint<DOUBLETYPE,SINGLETYPE,FRACTION_WIDTH> FP8;

#define MAXM(X, Y) (((X) > (Y)) ? (X) : (Y))

inline float addfloat(float a, float b){
	return a * b;
}

inline float addfixed(FP8 a, FP8 b){
	return a * b;
}


int main(int argc, char* argv[]){
	int number_random_tests = 1000000;
	double a = get_wall_time();	
	for(int i = 0; i < number_random_tests; i++){
		addfloat(static_cast<float>(rand())/static_cast<float>(RAND_MAX), static_cast<float>(rand())/static_cast<float>(RAND_MAX));
	}
	a = get_wall_time() - a;
	std::cout << "float time: " << a << " \n";
	
	double b = get_wall_time();	
	for(int i = 0; i < number_random_tests; i++){
		addfixed(static_cast<float>(rand())/static_cast<float>(RAND_MAX), static_cast<float>(rand())/static_cast<float>(RAND_MAX));
	}
	b = get_wall_time() - b;
	std::cout << "fixed time: " << b << " \n";

	std::cout << "factor: " << (b/a) << "\n";
}

