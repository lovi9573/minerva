/*
 * unittest_fixedpoint.cpp
 *
 *  Created on: Jul 29, 2015
 *      Author: jlovitt
 */


#define FIXED_POINT_FRACTION_WIDTH_N 14
#define FIXED_POINT_TYPE int16_t
#define FIXED_POINT_DOUBLE_WIDE_TYPE int32_t


#define FIXED_POINT
#include <stdint.h>
#include <assert.h>
#include <ostream>
#include <string>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <common/fixedpoint.h>
#include <math.h>
#include "unittest_main.h"



//typedef FixedPoint<DOUBLETYPE,SINGLETYPE,0> FP;
//typedef FixedPoint<DOUBLETYPE,SINGLETYPE,FRACTION_WIDTH> FP8;

#define MAXM(X, Y) (((X) > (Y)) ? (X) : (Y))


bool close(float a, float b, float eps){
	//printf("close: |%f - %f| = %f < %f\n",a,b,fabs(a-b),eps);
	return fabs(a - b) < eps;
}

float getEpsilon(int fracw){
	return 1.0f/((float)(1 << fracw));
}

float getMax(int fracw){
	return (float)(((FIXED_POINT_TYPE)1) << (sizeof(FIXED_POINT_TYPE)*8 - fracw -1)) \
			- getEpsilon(fracw);
}

float getMin(int fracw){
	return (float)(((FIXED_POINT_DOUBLE_WIDE_TYPE)1) << (sizeof(FIXED_POINT_TYPE)*8 - fracw -1)) ;
}

TEST(FixedPoint, AddMaxFracW){
	const int FRACW = 15;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	//float MIN = getMin(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	FP f0(a);
	FP f1(b);
	FP r = f0 + f1;

	EXPECT_NEAR((float)r ,a + b, 2*EPSILON) << " a: " << a << " b: "<< b;
	r = a;
	r += c;
	EXPECT_NEAR((float)r , a + c, 2*EPSILON)  << " a: " << a << " b: "<< c;
}

TEST(FixedPoint, AddMinFracW){
	const int FRACW = 1;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	//printf("MAX: %f", MAX);
	//float MIN = getMin(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	FP f0(a);
	//printf("a: %f\n",(float)f0);
	FP f1(b);
	//printf("b: %f\n",(float)f1);
	FP r = f0 + f1;

	EXPECT_NEAR((float)r ,a+b, 2*EPSILON) ;
	r = a;
	r += c;
	EXPECT_NEAR((float)r , a + c, 2*EPSILON);
}

TEST(FixedPoint, AddInt){
	const int FRACW = 0;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	//float MIN = getMin(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	FP f0(a);
	//printf("%f\n",(float)f0);
	FP f1(b);
	//printf("%f\n",(float)f1);
	FP r = f0 + f1;

	EXPECT_NEAR((float)r ,a+b, 2*EPSILON);
	r = a;
	r += c;
	EXPECT_NEAR((float)r , a + c, 2*EPSILON);
}


TEST(FixedPoint, SubMaxFracW){
	const int FRACW = 15;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	//float MIN = getMin(FRACW);

	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	FP f0(a);
	//printf("%f\n",(float)f0);
	FP f1(b);
	//printf("%f\n",(float)f1);
	FP r = f0 - f1;

	EXPECT_NEAR((float)r ,a-b, 2*EPSILON);
	r = a;
	r -= c;
	EXPECT_NEAR((float)r , a - c, 2*EPSILON);
}

TEST(FixedPoint, SubMinFracW){
	const int FRACW = 1;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	//float MIN = getMin(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	FP f0(a);
	//printf("%f\n",(float)f0);
	FP f1(b);
	//printf("%f\n",(float)f1);
	FP r = f0 - f1;

	EXPECT_NEAR((float)r ,a-b, 2*EPSILON);
	r = a;
	r -= c;
	EXPECT_NEAR((float)r , a - c, 2*EPSILON);
}

TEST(FixedPoint, SubInt){
	const int FRACW = 0;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX/2);
	FP f0(a);
	FP f1(b);
	FP r = f0 - f1;

	EXPECT_NEAR((float)r ,a-b, 2*EPSILON)  << " a: " << a << " b: "<< b;
	r = a;
	r -= c;
	EXPECT_NEAR((float)r , a - c, 2*EPSILON)  << " a: " << a << " b: "<< c;
}

/*
 * Multiply
 */


TEST(FixedPoint, MulMaxFracW){
	const int FRACW = 15;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	FP f0(a);
	FP f1(b);
	FP r = f0 * f1;
	EXPECT_NEAR((float)r , a*b, MAXM(EPSILON*a + EPSILON*b + EPSILON*EPSILON, 2*EPSILON) );
	r = a;
	r *= c;
	EXPECT_NEAR((float)r, a*c , MAXM(EPSILON*a + EPSILON*c + EPSILON*EPSILON, 2*EPSILON) );
}

TEST(FixedPoint, MulMinFracW){
	const int FRACW = 1;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	FP f0(a);
	FP f1(b);
	FP r = f0 * f1;
	EXPECT_NEAR((float)r , a*b, MAXM(EPSILON*a + EPSILON*b + EPSILON*EPSILON, 2*EPSILON) );
	r = a;
	r *= c;
	EXPECT_NEAR((float)r, a*c , MAXM(EPSILON*a + EPSILON*c + EPSILON*EPSILON, 2*EPSILON) );
}


TEST(FixedPoint, MulInt){
	const int FRACW = 0;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* sqrt(static_cast<float>(MAX));
	FP f0(a);
	FP f1(b);
	FP r = f0 * f1;
	EXPECT_NEAR((float)r , a*b, MAXM(EPSILON*a + EPSILON*b + EPSILON*EPSILON, 2*EPSILON) );
	r = a;
	r *= c;
	EXPECT_NEAR((float)r, a*c , MAXM(EPSILON*a + EPSILON*c + EPSILON*EPSILON, 2*EPSILON) );
}

/*
 * Division
 */

TEST(FixedPoint, DivMaxFracW){
	const int FRACW = 15;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* (static_cast<float>(MAX) - a/static_cast<float>(MAX)) + a/static_cast<float>(MAX);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* (static_cast<float>(MAX) - a/static_cast<float>(MAX)) + a/static_cast<float>(MAX);
	FP f0(a);
	FP f1(b);
	FP r = f0 / f1;
	//printf("%f\n",(float)r);
	EXPECT_NEAR((float)r ,a/b , MAXM(((a + EPSILON)/(b - EPSILON)) + EPSILON - (a/b), 2*EPSILON) );
	r = a;
	r /= c;
	EXPECT_NEAR((float)r, a/c  , MAXM(((a + EPSILON)/(c - EPSILON)) + EPSILON - (a/c), 2*EPSILON) );
}

TEST(FixedPoint, DivMinFracW){
	const int FRACW = 1;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* (static_cast<float>(MAX) - a/static_cast<float>(MAX)) + a/static_cast<float>(MAX);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* (static_cast<float>(MAX) - a/static_cast<float>(MAX)) + a/static_cast<float>(MAX);
	FP f0(a);
	FP f1(b);
	FP r = f0 / f1;
	//printf("%f\n",(float)r);
	EXPECT_NEAR((float)r ,a/b , MAXM(((a + EPSILON)/(b - EPSILON)) + EPSILON - (a/b), 2*EPSILON) );
	r = a;
	r /= c;
	EXPECT_NEAR((float)r, a/c  , MAXM(((a + EPSILON)/(c - EPSILON)) + EPSILON - (a/c), 2*EPSILON) );
}

TEST(FixedPoint, DivInt){
	const int FRACW = 0;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	float a = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* static_cast<float>(MAX);
	float b = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* (static_cast<float>(MAX) - a/static_cast<float>(MAX)) + a/static_cast<float>(MAX);
	float c = (static_cast<float>(rand())/ static_cast<float>(RAND_MAX))* (static_cast<float>(MAX) - a/static_cast<float>(MAX)) + a/static_cast<float>(MAX);
	FP f0(a);
	FP f1(b);
	FP r = f0 / f1;
	//printf("%f\n",(float)r);
	EXPECT_NEAR((float)r ,a/b , MAXM(((a + EPSILON)/(b - EPSILON)) + EPSILON - (a/b), 2*EPSILON) );
	r = a;
	r /= c;
	EXPECT_NEAR((float)r, a/c  , MAXM(((a + EPSILON)/(c - EPSILON)) + EPSILON - (a/c), 2*EPSILON) );
}


TEST(FixedPoint, Overflow){
	const int FRACW = 8;
	float EPSILON  = getEpsilon(FRACW);
	float MAX = getMax(FRACW);
	float min = getMin(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	FP f0((float)(1 << (sizeof(FIXED_POINT_TYPE)*8 - FRACW -1)));
	//printf("%f\n",(float)f0);
	FP f1((float)(1 << (sizeof(FIXED_POINT_TYPE)*8 - FRACW -1)));
	//printf("%f\n",(float)f1);
	FP r = f0 * f1;
	EXPECT_NEAR((float)r , MAX , EPSILON);
	FP f2(0.0f - (float)(1 << (sizeof(FIXED_POINT_TYPE)*8 - FRACW -1)));
	r = f2 * f1;
	//printf("%f\n",(float)r);
	EXPECT_NEAR((float)r , 0.0 - min, EPSILON);
	r.Report();
}

TEST(FixedPoint, DISABLED_Underflow){
	const int FRACW = 8;
	float EPSILON  = getEpsilon(FRACW);
	//float MAX = getMax(FRACW);
	//float min = getMin(FRACW);
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	FP f0(0.001f);
	FP f1(123.0f);
	FP r = f0 / f1;
	EXPECT_NEAR((float)r ,EPSILON, EPSILON/2);
	r /=  f1;
	EXPECT_NEAR((float)r ,EPSILON, EPSILON/2);
}

TEST(FixedPoint, Equality){
	const int FRACW = 8;
	typedef FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FRACW> FP;
	FP f0(0.13f);
	FP f1(0.13f);
	EXPECT_TRUE(f0 == f1);
	FP f2(0.000000013f);
	FP f3(0.000000013f);
	EXPECT_TRUE(f2 == f3);
}

TEST(FixedPoint, Print){
	FixedPoint<int32_t,int16_t,8> printtest(123.45678f);
	std::vector<std::string> expected = {"123.457","123.453"};
	if(std::find(expected.begin(), expected.end(), printtest.str()) != expected.end()){
		SUCCEED();
	}
	else{
		FAIL();
	}
	//EXPECT_STREQ("123.457",printtest.str());
}

