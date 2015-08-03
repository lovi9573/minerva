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

#define FRACTION_WIDTH 10
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
	FP8 f0(4.4);
	//printf("%f\n",(float)f0);
	FP8 f1(8.8);
	//printf("%f\n",(float)f1);
	FP8 r = f0 + f1;

	assert(close((float)r ,13.2, 2*EPSILON));
	r += 10.1;
	assert(close(23.3,(float)r , 2*EPSILON));
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
	FP8 f0(8.8);
	FP8 f1(4.3);
	FP8 r = f0 - f1;
	assert(close((float)r ,4.5, 2*EPSILON));
	r -= 1.1;
	assert(close(3.4,(float)r , 2*EPSILON));
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
	FP8 f0(4.4);
	FP8 f1(7.1);
	FP8 r = f0 * f1;
	assert(close((float)r ,4.4f*7.1f, EPSILON*4.4 + EPSILON*8.8 + EPSILON*EPSILON));
	r = 8.9f;
	r *= 0.13;
	assert(close((float)r, 8.9f*0.13f , EPSILON*3.1f + EPSILON*38.72f + EPSILON*EPSILON));
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
	float a = 12.6f;
	float b = 4.8f;
	float c = 1.2f;
	FP8 f0(a);
	FP8 f1(b);
	FP8 r = f0 / f1;
	//printf("%f\n",(float)r);
	assert(close((float)r ,a/b , MAXM(((a + EPSILON)/(b - EPSILON)) - (a/b), EPSILON) ));
	r = a;
	r /= c;
	assert(close((float)r, a/c  , MAXM(((a + EPSILON)/(c - EPSILON)) - (a/c), EPSILON) ));
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
	FP8 f0(0.05f);
	FP8 f1(87.5f);
	FP8 r = f0 / f1;
	assert(close((float)r ,EPSILON, EPSILON/2));
	r /=  f1;
	assert(close((float)r ,EPSILON, EPSILON/2));
}



int main(int argc, char* argv[]){
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
	printf("PASS!\n");
}

