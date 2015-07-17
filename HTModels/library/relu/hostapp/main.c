#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

extern void relu_forward(int32_t *input_q88_data, int32_t *output_q88_data, size_t numbers );
void usage (char *);

int main(int argc, char **argv)
{

    uint32_t i;
    uint32_t vecLen;
    int32_t *a1, *a2;

    // check command line args
    if (argc == 1)
	vecLen = 100;         // default vecLen
    else if (argc == 2) {
	vecLen = atoi(argv[1]);
	if (vecLen > 0) {
	    printf("Running UserApp.exe with vecLen = %llu\n", (long long) vecLen);
	} else {
	    usage (argv[0]);
	    return 0;
	}
    } else {
	usage (argv[0]);
	return 0;
    }

    a1 = (int32_t *) malloc(vecLen*sizeof(int32_t));
    a2 = (int32_t *) malloc(vecLen*sizeof(int32_t));

    for (i = 0; i < vecLen; i++) {
    	a1[i] = i-vecLen/2;
    }

    relu_forward(a1, a2, vecLen);
    printf("Relu done\n");

    // check results
    int err_cnt = 0;

    for (i = 0; i < vecLen; i++) {
    	if(a1[i] > 0){
			if (a2[i] != a1[i]) {
				printf("a3[%llu] is %llu, should be %llu\n",
				(long long)i, (long long)a2[i], (long long)a1[i]);
				err_cnt++;
			}
    	}else{
			if (a2[i] != 0) {
				printf("a3[%llu] is %llu, should be %llu\n",
				(long long)i, (long long)a2[i], (long long)0);
				err_cnt++;
			}
    	}
    }

    if (err_cnt)
	printf("FAILED: detected %d issues!\n", err_cnt);
    else
	printf("PASSED\n");

    return err_cnt;
}

// Print usage message and exit with error.
void
usage (char* p)
{
    printf("usage: %s [count (default 100)] \n", p);
    exit (1);
}

