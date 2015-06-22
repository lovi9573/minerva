

/*
* Forward convolution with 
* 256 image batch, 
* 11x11 filter
* 96 filters
* 4 stride
* 227x227x3 image dimensions
*
*/

__kernel void convfwd(	_global float* images, 
						_global float* filters,
						_global float* targets,
						int imgY,
						int imgX,
						int imgC,
						int imgN,
						int filterY,
						int filterX,
						int filterN,
						int stride,
						int padding){
	int numImages = 256;
	int imgsPerThread = 4; //numImages % 128
	int numFilterColors = imgC;
	int numFilters = filterN;
//From filterActs_YxX_color template	
	int Block_height_y = 4; //block dims
	int Block_width_x = 32; // block dims
	int imgsPerThread = 4;
	int filtersPerThread = 8;
	//int numColors;
	int pixelCache = 4;
	//bool scale = false;
	//bool checkImgBounds = false;
	
	//Preload arrays
	__local float shFilters[pixelCache*imgC][Block_height_y*filtersPerThread];
	__local float shImages[pixelCache*imgC][Block_width_x * imgsPerThread];
	
	//Pixel counts
	const int imgPixels = imgY*imgX;
	const int filterPixels = filterY * filterX;
	
	//
    const int blocksPerModule = numFilters / (Block_height_y*filtersPerThread);
    const int moduleIdx = 0; //get_group_id(1) / blocksPerModule;  
    const int blockFilterIdx = get_group_id(1) % blocksPerModule;
	
	// Worker indexing
	//TODO: substitute get_global_id()
	const int thread_id_within_block_flat = get_local_id(1) * Block_width_x + get_local_id(0);
    const int imgLoadModPosY = padding; 
    const int imgLoadModPosX = padding;
    const int numModules = 1; //numModulesY * numModulesX;
    const int shFilterLoadY = thread_id_within_block_flat / (Block_height_y * filtersPerThread);
    const int shFilterLoadX = thread_id_within_block_flat % (Block_height_y * filtersPerThread);
    const int myImgIdx = get_group_id(0) * Block_width_x * imgsPerThread + get_local_id(0);
    
    images += myImgIdx;  //pointer arithmetic.
    filters += filtersPerThread * Block_height_y * blockFilterIdx
             + shFilterLoadY * numFilters + shFilterLoadX;
    targets += moduleIdx * numImages
            + (blockFilterIdx * Block_height_y * filtersPerThread + get_local_id(1)*filtersPerThread) * numImages 
            + myImgIdx; //Line 2 of this statement: * numModulesY * numModulesX
    float prod[filtersPerThread][imgsPerThread];
	#pragma unroll
	for(int f = 0; f < filtersPerThread; f++) {
		#pragma unroll
		for(int g = 0; g < imgsPerThread; g++) {
			prod[f][g] = 0;
		}
	}  
    for (int p = 0; p < filterPixels; p += pixelCache) {
        /*
         * Load pixelCache pixels from Block_height_y*filtersPerThread filters
         * This condition covers the case when Block_width_x is not divisible by filtersPerThread.
         * In this case, not all of the threads will participate in the loading operation.
         * This ensures that in each loop iteration, an integer number of rows of shFilters
         * are filled, which makes indexing simple.
         */
        if (Block_width_x % filtersPerThread == 0 || shFilterLoadY < Block_width_x/filtersPerThread) {
            #pragma unroll
            for (int p2 = 0; p2 < pixelCache; p2 += Block_width_x/filtersPerThread) {
                const bool omit = pixelCache % (Block_width_x / filtersPerThread) == 0;
                const int preloadPx = shFilterLoadY + p2;
                if (omit || preloadPx < pixelCache) {
                    if (p + preloadPx < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = 0;
                        }
                    }
                }
            }
        }

        /*
         * Load pixelCache pixels from Block_width_x*imgsPerThread images.
         */
        #pragma unroll
        for (int ly = 0; ly < pixelCache; ly += Block_height_y) {
            const int preloadPx = ly + threadIdx.y;
            const int pixIdx = p + preloadPx;
            const bool omit = pixelCache % Block_height_y == 0; // Compile-time condition
            /*
             * Don't load any image pixels corresponding to filter pixels that don't exist.
             */
            if (pixIdx < filterPixels && (omit || preloadPx < pixelCache)) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + pixIdx / filterSize;

                if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    float* m = &images[imgStride * (y * imgSizeX + x)];

                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            if (!checkImgBounds || myImgIdx + i * Block_width_x < numImages) {
                                shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = m[c * imgStride * imgPixels + i * Block_width_x];
                            } else {
                                shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
                        }
                    }
                }
            }
        }

        barrier();

        #pragma unroll
        for (int i = 0; i < pixelCache*numColors; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
                    prod[f][g] += shImages[i][g + threadIdx.x * imgsPerThread] * shFilters[i][threadIdx.y * filtersPerThread + f];
                }
            }
        }
        barrier();
    }

    #pragma unroll
	for (int g = 0; g < imgsPerThread; g++) {
		if (!checkImgBounds || myImgIdx + g * Block_width_x < numImages) {
			#pragma unroll
			for (int f = 0; f < filtersPerThread; f++) {
				targets[g * Block_width_x + f * numImages * numModules] = scaleOutputs * prod[f][g];
			}
		}
	}
    
    
    
}