#include <Ht.h>
using namespace Ht;

void conv_forward(void *input_q88_data, size_t num_img, size_t img_dim, size_t img_channels, size_t img_alloc,
		 void *filters_q88_data, size_t num_filters, size_t filter_dim, size_t stride, size_t filter_alloc,
		 void *output_q88_data, size_t output_alloc,
		 uint16_t fraction_width){

	// Get interfaces
	CHtHif *pHtHif = new CHtHif();
	int unitCnt = pHtHif->GetUnitCnt();
	printf("#AUs = %d\n", unitCnt);

	CHtAuUnit ** pAuUnits = new CHtAuUnit * [unitCnt];
	for (int unit = 0; unit < unitCnt; unit++){
		pAuUnits[unit] = new CHtAuUnit(pHtHif);
	}

	// Set up memory
	uint64_t* ht_input_img_data = (uint64_t*)pHtHif->MemAlloc(img_alloc/4* sizeof(uint64_t));
	uint64_t* ht_input_filter_data = (uint64_t*)pHtHif->MemAlloc(filter_alloc/4* sizeof(uint64_t));
	uint64_t* ht_output_data = (uint64_t*)pHtHif->MemAlloc(output_alloc/4 * sizeof(uint64_t));
	if(!ht_input_img_data || !ht_input_filter_data || !ht_output_data){
		printf("HT MemAlloc failed.\n");
		exit(1);
	}
	pHtHif->MemCpy(ht_input_img_data, input_q88_data, img_alloc/4 * sizeof(uint64_t));
	pHtHif->MemCpy(ht_input_filter_data, filters_q88_data, filter_alloc/4 * sizeof(uint64_t));

	// avoid bank aliasing for performance
	if (unitCnt > 16 && !(unitCnt & 1)) unitCnt -= 1;
	printf("stride = %d\n", unitCnt);
	fflush(stdout);

	// Initialize modules with messages
	pHtHif->SendAllHostMsg(IMG_ADDR, (uint64_t)ht_input_img_data);
	pHtHif->SendAllHostMsg(IMG_NUM, (uint64_t)num_img);
	pHtHif->SendAllHostMsg(IMG_DIM, (uint64_t)img_dim);
	pHtHif->SendAllHostMsg(IMG_CHANNELS, (uint64_t)img_channels);
	pHtHif->SendAllHostMsg(FILTER_ADDR, (uint64_t)ht_input_filter_data);
	pHtHif->SendAllHostMsg(FILTER_NUM, (uint64_t)num_filters);
	pHtHif->SendAllHostMsg(FILTER_DIM, (uint64_t)filter_dim);
	pHtHif->SendAllHostMsg(STRIDE, (uint64_t)stride);
	pHtHif->SendAllHostMsg(OUT_ADDR, (uint64_t)ht_output_data);
	pHtHif->SendAllHostMsg(FRACTION_WIDTH, (uint64_t)fraction_width);


	// Send calls to units
	for (int unit = 0; unit < unitCnt; unit++)
		pAuUnits[unit]->SendCall_htmain(unit /*offset*/, unitCnt /*stride*/);

	// Wait for returns
	for (int unit = 0; unit < unitCnt; unit++) {
		while (!pAuUnits[unit]->RecvReturn_htmain())
			usleep(1000);
	}

	//Copy results out.
	pHtHif->MemCpy(output_q88_data, ht_output_data, output_alloc/4 * sizeof(uint64_t));

}

