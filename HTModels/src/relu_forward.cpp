#include <Ht.h>
using namespace Ht;

void relu_forward(int64_t *input_q88_data, int64_t *output_q88_data, size_t numbers)
{

	// Get interfaces
	CHtHif *pHtHif = new CHtHif();
	int unitCnt = pHtHif->GetUnitCnt();
	printf("#AUs = %d\n", unitCnt);

	CHtAuUnit ** pAuUnits = new CHtAuUnit * [unitCnt];
	for (int unit = 0; unit < unitCnt; unit++){
		pAuUnits[unit] = new CHtAuUnit(pHtHif);
	}

	// Set up memory
	int64_t* ht_input_data = (int64_t*)pHtHif->MemAlloc(numbers * sizeof(int64_t));;
	int64_t* ht_output_data = (int64_t*)pHtHif->MemAlloc(numbers * sizeof(int64_t));
	if(!ht_input_data){
		printf("HT MemAlloc failed.\n");
		exit(1);
	}
	pHtHif->MemCpy(ht_input_data, input_q88_data, numbers * sizeof(int64_t));

	// avoid bank aliasing for performance
	if (unitCnt > 16 && !(unitCnt & 1)) unitCnt -= 1;
	printf("stride = %d\n", unitCnt);
	fflush(stdout);

	// Initialize modules with messages
	printf("Device Allocated input data at %lu\n",(uint64_t)ht_input_data);
	pHtHif->SendAllHostMsg(IN_ADDR, (int64_t)ht_input_data);
	pHtHif->SendAllHostMsg(OUT_ADDR, (int64_t)ht_output_data);
	pHtHif->SendAllHostMsg(VEC_LEN, (uint64_t)numbers);

	// Send calls to units
	for (int unit = 0; unit < unitCnt; unit++)
		pAuUnits[unit]->SendCall_htmain(unit /*offset*/, unitCnt /*stride*/);

	// Wait for returns
	for (int unit = 0; unit < unitCnt; unit++) {
		while (!pAuUnits[unit]->RecvReturn_htmain())
			usleep(1000);
	}

	//Copy results out.
	pHtHif->MemCpy(output_q88_data, ht_output_data, numbers * sizeof(int64_t));

}

