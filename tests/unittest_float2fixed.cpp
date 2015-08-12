#include "unittest_main.h"
#include <op/impl/fpga/fpga_util.h>

using namespace minerva;
using namespace std;

/*

TEST(float2q88, q882float) {
	int n = 5;
	float f[n];
	float f_inp[] = {-2.1, 1.1, 2.22, 127.333, 500.12341234};
	float f_exp[] = {-2.1, 1.1, 2.22, 127.333, 127.996};
	char q[2*n];
	char q_exp[] = {~0x18,~0x02,25,1,56,2, 85,127, 0xff,0x7f};

	float2qxx(f_inp, q, n, 16,8);
	for (int i = 0; i < 2*n; ++i) {
		EXPECT_EQ(q_exp[i], q[i]);
	}

	qxx2float(q_exp,f,n,16,8);
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(f_exp[i], f[i], 1.0/256);
	}
}


TEST(float2q214, q2142float) {
	int n = 4;
	float f[n];
	float f_exp[] = {-4.4,1.1, 2.22, 3.333333};
	char q[2*n];
	char q_exp[] = {1638,1,3604,2,5461,3};

	float2qxx(f_exp, q, n, 16,12);
	for (int i = 0; i < 2*n; ++i) {
		//EXPECT_EQ(q_exp[i], q[i]);
	}

	qxx2float(q,f,n,16,12);
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(f_exp[i], f[i], 1.0/4096);
	}
}
*/
