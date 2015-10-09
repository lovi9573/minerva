#include <Ht.h>
using namespace Ht;

#include "../HTModels.h"

size_t get64_bit_aligned(size_t size){
		return ((size+7)/8)*8;
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
/*
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
		pAuUnits[unit]->SendCall_htmain(unit /*offset*/, unitCnt /*stride*/, CONV_FORWARD);

	// Wait for returns
	for (int unit = 0; unit < unitCnt; unit++) {
		while (!pAuUnits[unit]->RecvReturn_htmain())
			usleep(1000);
	}

	//Copy results out.
	pHtHif->MemCpy(output_q88_data, ht_output_data, output_alloc);

		pHtHif->MemFree(ht_input_img_data);
		pHtHif->MemFree(ht_input_filter_data);
		pHtHif->MemFree(ht_output_data);

	for (int unit = 0; unit < unitCnt; unit++){
		delete pAuUnits[unit];
	}
	delete pHtHif;
}

void conv_backward_data_ht(void* top_diff, size_t top_alloc,
							void* filter_data, int num_filters, int filter_dim, int stride, int pad_x, int pad_y, size_t filter_alloc,
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
	uint64_t* ht_top_diff_data = (uint64_t*)pHtHif->MemAlloc(top_alloc_ht);
	uint64_t* ht_filter_data = (uint64_t*)pHtHif->MemAlloc(filter_alloc_ht);
	uint64_t* ht_bottom_diff_data = (uint64_t*)pHtHif->MemAlloc(bottom_alloc_ht);
	if(!ht_top_diff_data || !ht_filter_data || !ht_bottom_diff_data){
		printf("HT MemAlloc failed.\n");
		exit(1);
	}
	pHtHif->MemCpy(ht_top_diff_data, top_diff, top_alloc_ht);
	pHtHif->MemCpy(ht_filter_data, filter_data, filter_alloc_ht);

	// avoid bank aliasing for performance
	if (unitCnt > 16 && !(unitCnt & 1)) unitCnt -= 1;
	printf("stride = %d\n", unitCnt);
	fflush(stdout);


	// Initialize modules with messages
	pHtHif->SendAllHostMsg(TOP_ADDR, (uint64_t)ht_top_diff_data);
	pHtHif->SendAllHostMsg(FILTER_ADDR, (uint64_t)ht_filter_data);
	pHtHif->SendAllHostMsg(BOTTOM_ADDR, (uint64_t)ht_bottom_diff_data);
	pHtHif->SendAllHostMsg(BOTTOM_SAMPLES, (uint64_t)bottom_samples);
	pHtHif->SendAllHostMsg(BOTTOM_DIM, (uint64_t)bottom_width);
	pHtHif->SendAllHostMsg(BOTTOM_CHANNELS, (uint64_t)bottom_channels);
	pHtHif->SendAllHostMsg(NUM_FILTERS, (uint64_t)num_filters);
	pHtHif->SendAllHostMsg(FILTER_DIM, (uint64_t)filter_dim);
	pHtHif->SendAllHostMsg(STRIDE_X, (uint64_t)stride);
	pHtHif->SendAllHostMsg(STRIDE_Y, (uint64_t)stride);
	//pHtHif->SendAllHostMsg(FRACTION_WIDTH, (uint64_t)frac_w);
	pHtHif->SendAllHostMsg(PAD_X, (uint64_t)pad_x);
	pHtHif->SendAllHostMsg(PAD_Y, (uint64_t)pad_y);
	pHtHif->SendAllHostMsg(DATA_STRIDE_Y, (uint64_t)(2*bottom_width));
	pHtHif->SendAllHostMsg(DATA_STRIDE_C, (uint64_t)(2*bottom_width*bottom_width));
	pHtHif->SendAllHostMsg(DATA_STRIDE_S, (uint64_t)(2*bottom_width*bottom_width*bottom_channels));


	// Send calls to units
	for (int unit = 0; unit < unitCnt; unit++)
		pAuUnits[unit]->SendCall_htmain(unit , unitCnt, CONV_BACKWARD_DATA );

	// Wait for returns
	for (int unit = 0; unit < unitCnt; unit++) {
		while (!pAuUnits[unit]->RecvReturn_htmain())
			usleep(1000);
	}

	//Copy results out.
	pHtHif->MemCpy(bottom_diff, ht_bottom_diff_data, bottom_alloc);

}


void ConvBackwardBias_ht(void* top_diff, size_t top_elements, size_t top_column_stride, size_t top_channel_stride, size_t top_image_stride,
						void* bottom_diff, size_t bottom_alloc, int channels,
						int frac_w){
	printf("Entry to HT interface\n");
	// Get interfaces
	CHtHif *pHtHif = new CHtHif();
	int unitCnt = pHtHif->GetUnitCnt();
	printf("#AUs = %d\n", unitCnt);

	printf("Allocating HT units\n");
	CHtAuUnit ** pAuUnits = new CHtAuUnit * [unitCnt];
	for (int unit = 0; unit < unitCnt; unit++){
		pAuUnits[unit] = new CHtAuUnit(pHtHif);
	}

	//Ensure that the allocation can be interpreted as uint64_t
	size_t top_alloc_ht = get64_bit_aligned(top_elements*2);
	size_t bottom_alloc_ht = get64_bit_aligned(bottom_alloc);

	// Set up memory
	printf("Memalloc on HT\n");
	uint64_t* ht_top_diff_data = (uint64_t*)pHtHif->MemAlloc(top_alloc_ht);
	uint64_t* ht_bottom_diff_data = (uint64_t*)pHtHif->MemAlloc(bottom_alloc_ht);
	if(!ht_top_diff_data || !ht_bottom_diff_data){
		printf("HT MemAlloc failed.\n");
		exit(1);
	}

/*	printf("top alloc bytes %d\n",top_elements*2);
    for(int i =0; i < (int)top_elements; i+=8){
    	printf(" %lx ",*((uint64_t*)(top_diff+i)));
    }*/

	pHtHif->MemCpy(ht_top_diff_data, top_diff, top_elements*2);

	// avoid bank aliasing for performance
	if (unitCnt > 16 && !(unitCnt & 1)) unitCnt -= 1;
	printf("stride = %d\n", unitCnt);
	fflush(stdout);


	// Initialize modules with messages
	pHtHif->SendAllHostMsg(TOP_ADDR, (uint64_t)ht_top_diff_data);
	pHtHif->SendAllHostMsg(BIAS_ADDR, (uint64_t)ht_bottom_diff_data);
	pHtHif->SendAllHostMsg(CHANNELS, (uint64_t)channels);
	pHtHif->SendAllHostMsg(CHANNEL_STRIDE, (uint64_t)top_channel_stride);
	pHtHif->SendAllHostMsg(SIZE, (uint64_t)top_elements);

	// Send calls to units
	printf("Send call to HT\n");
	for (int unit = 0; unit < unitCnt; unit++)
		pAuUnits[unit]->SendCall_htmain(unit , unitCnt, CONV_BACKWARD_BIAS );

	// Wait for returns
	for (int unit = 0; unit < unitCnt; unit++) {
		while (!pAuUnits[unit]->RecvReturn_htmain())
			usleep(1000);
	}

	//Copy results out.
	pHtHif->MemCpy(bottom_diff, ht_bottom_diff_data, bottom_alloc+20);
/*    for(int i =0; i < (int)bottom_alloc+20; i++){
    	printf(" %d ",((uint16_t*)bottom_diff)[i]);
    }*/


}
/*
	ConvBackwardFilter_ht(top_diff,top_size,
				 bottom, bottom_size, bottom_column_stride, bottom_channel_stride, bottom_image_stride,
				 filter_diff, filter_size, filter_column_stride, filter_channel_stride, filter_element_stride,
				 frac_width,
				 FIXED_POINT_FRACTION_WIDTH){

	}
*/


