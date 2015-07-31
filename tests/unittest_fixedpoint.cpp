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

typedef FixedPoint<int32_t,int16_t,0> FP;
typedef FixedPoint<int32_t,int16_t,8> FP4;

void addi(){
	FP f0(4);
	FP f1(8);
	FP r = f0 + f1;
	assert(r == 12);
	r += 10;
	printf("%d\n)",r);
	assert(r == 22);
}

void addf(){
	FP4 f0(4.4);
	FP4 f1(8.8);
	FP4 r = f0 + f1;
	//assert((float)r.value == 13.2);
	r += 10.1;
	printf("%f\n)",(float)r);
	assert(23.3 - (float)r < 0.01);
}


int main(int argc, char* argv[]){
	addi();
	addf();
}

