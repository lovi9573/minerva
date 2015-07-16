#include "unittest_main.h"
#include <op/impl/fpga/fpga_util.h>

using namespace minerva;
using namespace std;


TEST(float2q88, q882float) {
	float f[3];
	float f_exp[] = {1.1, 2.22, 3.333};
	char q[6];
	char q_exp[] = {25,1,56,2,85,3};

	float2qxx(f_exp, q, 3, 16,8);
	for (int i = 0; i < 6; ++i) {
		EXPECT_EQ(q_exp[i], q[i]);
	}

	qxx2float(q_exp,f,3,16,8);
	for (int i = 0; i < 3; ++i) {
		EXPECT_NEAR(f_exp[i], f[i], 1.0/256);
	}
}


TEST(float2q214, q2142float) {
	float f[3];
	float f_exp[] = {1.1, 2.22, 3.333333};
	char q[6];
	char q_exp[] = {1638,1,3604,2,5461,3};

	float2qxx(f_exp, q, 3, 16,14);
	for (int i = 0; i < 6; ++i) {
		EXPECT_EQ(q_exp[i], q[i]);
	}

	qxx2float(q_exp,f,3,16,14);
	for (int i = 0; i < 3; ++i) {
		EXPECT_NEAR(f_exp[i], f[i], 1.0/16384);
	}
}
