#include <Ht.h>
using namespace Ht;


size_t get64_bit_aligned(size_t size){
	if (size % 8 != 0){
		return (size/8+1)*8;
	}
	return size;
}


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

	//Ensure that the allocation can be interpreted as uint64_t
	size_t img_alloc_ht = get64_bit_aligned(img_alloc);
	size_t filter_alloc_ht = get64_bit_aligned(filter_alloc);
	size_t output_alloc_ht = get64_bit_aligned(output_alloc);

	// Set up memory
	uint64_t* ht_input_img_data = (uint64_t*)pHtHif->MemAlloc(img_alloc_ht);
	uint64_t* ht_input_filter_data = (uint64_t*)pHtHif->MemAlloc(filter_alloc_ht);
	uint64_t* ht_output_data = (uint64_t*)pHtHif->MemAlloc(output_alloc_ht);
	if(!ht_input_img_data || !ht_input_filter_data || !ht_output_data){
		printf("HT MemAlloc failed.\n");
		exit(1);
	}
	pHtHif->MemCpy(ht_input_img_data, input_q88_data, img_alloc_ht);
	pHtHif->MemCpy(ht_input_filter_data, filters_q88_data, filter_alloc_ht);

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
	pHtHif->MemCpy(output_q88_data, ht_output_data, output_alloc);
}

void conv_backward_data_ht(void* top_diff, size_t top_alloc,
							void* filter_data, size_t num_filters, size_t filter_dim, size_t stride, size_t filter_alloc,
							void* bottom_diff, int bottom_width, int bottom_height, int bottom_channels, int bottom_samples, size_t bottom_alloc,
							int frac_w ){

	// Get interfaces
	CHtHif *pHtHif = new CHtHif();
	int unitCnt = pHtHif->GetUnitCnt();
	printf("#AUs = %d\n", unitCnt);

	CHtAuUnit ** pAuUnits = new CHtAuUnit * [unitCnt];
	for (int unit = 0; unit < unitCnt; unit++){
		pAuUnits[unit] = new CHtAuUnit(pHtHif);
	}

	//Ensure that the allocation can be interpreted as uint64_t
	size_t top_alloc_ht = get64_bit_aligned(top_alloc);
	size_t filter_alloc_ht = get64_bit_aligned(filter_alloc);
	size_t bottom_alloc_ht = get64_bit_aligned(bottom_alloc);

	// Set up memory
	uint64_t* ht_input_top_data = (uint64_t*)pHtHif->MemAlloc(top_alloc_ht);
	uint64_t* ht_input_filter_data = (uint64_t*)pHtHif->MemAlloc(filter_alloc_ht);
	uint64_t* ht_bottom_data = (uint64_t*)pHtHif->MemAlloc(output_alloc_ht);
	if(!ht_input_img_data || !ht_input_filter_data || !ht_output_data){
		printf("HT MemAlloc failed.\n");
		exit(1);
	}
	pHtHif->MemCpy(ht_input_top_data, top_diff, top_alloc_ht);
	pHtHif->MemCpy(ht_input_filter_data, filter_data, filter_alloc_ht);

	// avoid bank aliasing for performance
	if (unitCnt > 16 && !(unitCnt & 1)) unitCnt -= 1;
	printf("stride = %d\n", unitCnt);
	fflush(stdout);

	/*
	 * TODO(jesselovitt): change for backward_data

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
*/

	// Send calls to units
	for (int unit = 0; unit < unitCnt; unit++)
		pAuUnits[unit]->SendCall_htmain(unit /*offset*/, unitCnt /*stride*/);

	// Wait for returns
	for (int unit = 0; unit < unitCnt; unit++) {
		while (!pAuUnits[unit]->RecvReturn_htmain())
			usleep(1000);
	}

	//Copy results out.
	pHtHif->MemCpy(bottom_diff, ht_bottom_data, bottom_alloc);
}

