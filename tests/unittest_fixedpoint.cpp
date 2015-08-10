/*
 * unittest_fixedpoint.cpp
 *
 *  Created on: Jul 29, 2015
 *      Author: jlovitt
 */




#define FIXED_POINT
#include <stdint.h>
#include <assert.h>
#include <ostream>
#include <string>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <fixedpoint.h>
#include <math.h>

//TODO(jesse lovitt) Does not work for fractional widths > 8

#define FRACTION_WIDTH 14
#define SINGLETYPE int16_t
#define DOUBLETYPE int32_t

typedef FixedPoint<DOUBLETYPE,SINGLETYPE,0> FP;
typedef FixedPoint<DOUBLETYPE,SINGLETYPE,FRACTION_WIDTH> FP8;

#define MAXM(X, Y) (((X) > (Y)) ? (X) : (Y))

float EPSILON  = (1.0f/((float)(1 << FRACTION_WIDTH) - 1));
float MAX = (float)(
					((SINGLETYPE)1) << (sizeof(SINGLETYPE)*8 - FRACTION_WIDTH )
					- 1)
			- EPSILON;

float MIN = (float)(
					((DOUBLETYPE)1) << (sizeof(SINGLETYPE)*8 - FRACTION_WIDTH )
					-1);


bool close(float a, float b, float eps){
	//printf("close: |%f - %f| = %f < %f\n",a,b,fabs(a-b),eps);
	return fabs(a - b) < eps;
}

void addi(){
	FP f0(4);
	FP f1(8);
	FP r = f0 + f1;
	assert(r == 12);
	r += 10;
	//printf("%d\n)",r);
	assert(r == 22);
}

void addf(){
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	FP8 f0(a);
	//printf("%f\n",(float)f0);
	FP8 f1(b);
	//printf("%f\n",(float)f1);
	FP8 r = f0 + f1;

	assert(close((float)r ,a+b, 2*EPSILON));
	r = a;
	r += c;
	assert(close((float)r , a + c, 2*EPSILON));
}

void subi(){
	FP f0(8);
	FP f1(4);
	FP r = f0 - f1;
	assert(r == 4);
	r -= 2;
	//printf("%d\n)",r);
	assert(r == 2);
}

void subf(){
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX);
	FP8 f0(a);
	FP8 f1(b);
	FP8 r = f0 - f1;
	assert(close((float)r , a-b, 2*EPSILON));
	r = a;
	r -= c;
	assert(close((float)r, a-c , 2*EPSILON));
}


void muli(){
	FP f0(6);
	FP f1(8);
	FP r = f0 * f1;
	assert(r == 48);
	r *= 2;
	//printf("%d\n)",r);
	assert(r == 96);
}

void mulf(){
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	FP8 f0(a);
	FP8 f1(b);
	FP8 r = f0 * f1;
	assert(close((float)r , a*b, MAXM(EPSILON*a + EPSILON*b + EPSILON*EPSILON, 2*EPSILON) ));
	r = a;
	r *= c;
	assert(close((float)r, a*c , MAXM(EPSILON*a + EPSILON*c + EPSILON*EPSILON, 2*EPSILON) ));
}

void divi(){
	FP f0(32);
	FP f1(4);
	FP r = f0 / f1;
	assert(r == 8);
	r /= 4;
	//printf("%d\n)",r);
	assert(r == 2);
}

void divf(){
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* (static_cast<float>(MAX) - a/static_cast<float>(MAX)) + a/static_cast<float>(MAX);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* (static_cast<float>(MAX) - a/static_cast<float>(MAX)) + a/static_cast<float>(MAX);
	FP8 f0(a);
	FP8 f1(b);
	FP8 r = f0 / f1;
	//printf("%f\n",(float)r);
	assert(close((float)r ,a/b , MAXM(((a + EPSILON)/(b - EPSILON)) - (a/b), 2*EPSILON) ));
	r = a;
	r /= c;
	assert(close((float)r, a/c  , MAXM(((a + EPSILON)/(c - EPSILON)) - (a/c), 2*EPSILON) ));
}

void overflow(){
	FP8 f0((float)(1 << (sizeof(SINGLETYPE)*8 - FRACTION_WIDTH -1)));
	//printf("%f\n",(float)f0);
	FP8 f1((float)(1 << (sizeof(SINGLETYPE)*8 - FRACTION_WIDTH -1)));
	//printf("%f\n",(float)f1);
	FP8 r = f0 * f1;
	assert(close((float)r , MAX , EPSILON));
	r = -f0 * f1;
	//printf("%f\n",(float)r);
	assert(close((float)r , 0.0 - MIN, EPSILON));
}

void underflow(){
	FP8 f0(0.000005f);
	FP8 f1(87.5f);
	FP8 r = f0 / f1;
	assert(close((float)r ,EPSILON, EPSILON/2));
	r /=  f1;
	assert(close((float)r ,EPSILON, EPSILON/2));
}



int main(int argc, char* argv[]){
	int number_random_tests = 100;
	for(int i = 0; i < number_random_tests; i++){
		addi();
		addf();
		subi();
		subf();
		muli();
		mulf();
		divi();
		divf();
		overflow();
		underflow();
	}
	FixedPoint<int32_t,int16_t,8> printtest(123.45678f);
	printf("123.45678 == %s\n",printtest.str());
	printf("PASS!\n");
}

