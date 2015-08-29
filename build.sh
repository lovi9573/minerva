#!/bin/bash

source configure.in

function unknown_option {
  echo "Unknown option: $1"
  echo "Run ./configure --help to get help"
  exit 1
}

function run_clean {
  read -n 1 -s -p "Are you sure to delete builds? [y/n]
" yesorno;
  if [ "$yesorno" == "y" ]; then
      echo "Removing builds"
      rm -rf $BUILD_DIR
      python setup.py clean --all
      rm -rf build
      rm -rf dist
      rm -rf owl/libowl.so
  fi
  exit 0
}


CXXFLAGS="$CXXFLAGS \
  -DCUDA_ROOT=$CUDA_ROOT \
  -DCUDNN_ROOT=$CUDNN_ROOT \
  -DMPI_ROOT=$MPI_ROOT \
  -DBUILD_CXX_APPS=$BUILD_CXX_APPS \
  -DUSE_PROFILER=$USE_PROFILER \
  -DBUILD_TESTS=$BUILD_TESTS \
  -DBUILD_WITH_PS=$BUILD_WITH_PS \
  -DBUILD_CPU_ONLY=$BUILD_CPU_ONLY \
  -DBUILD_WITH_BLAS=$BUILD_WITH_BLAS \
  -DBUILD_WITH_MPI=$BUILD_WITH_MPI \
  -DBUILD_WITH_FPGA=$BUILD_WITH_FPGA \
  -DBLAS_ROOT=$BLAS_ROOT \
  -DHT_MODEL_TYPE=$HT_MODEL_TYPE \
  -DFIXED_POINT=$FIXED_POINT \
  -DFIXED_POINT_DOUBLE_WIDE_TYPE=$FIXED_POINT_DOUBLE_WIDE_TYPE \
  -DFIXED_POINT_TYPE=$FIXED_POINT_TYPE \
  -DFIXED_POINT_WORD_LENGTH_N=$FIXED_POINT_WORD_LENGTH_N \
  -DFIXED_POINT_FRACTION_WIDTH_N=$FIXED_POINT_FRACTION_WIDTH_N \
  -DPS_ROOT=$PS_ROOT \
  "

while [[ $# -gt 0 ]]; do
  case $1 in
    --help | -h)
      WANT_HELP=yes ;;
    -c | --clean)
      run_clean ;;
    -D)
      CXXFLAGS="$CXXFLAGS -D$2=ON"; shift ;;
    -D*)
      CXXFLAGS="$CXXFLAGS -D`expr "x$1" : "x-D\(.*\)"`" ;;
    *)
      unknown_option $1 ;;
  esac
  shift
done

#================================ help description =============================
if test "x$WANT_HELP" = xyes; then
    cat <<EOF
Usage: ./configure [OPTION]...
Configurations:
    -h, --help              Display this help and exit
    -c, --clean             Clean up debug and release build
    -Dvar=value | -D value  Directly specify definitions to be passed to CMake in cmdline
EOF
exit 0;
fi
#===============================================================================

#================================ main configuration =============================
if [ ! -d $BUILD_DIR ]; then
  mkdir $BUILD_DIR
fi
cd $BUILD_DIR
if [ $BUILD_WITH_FPGA -eq 1 ]; then
	cp -r ../HTModels . && make -C HTModels lib$HT_MODEL_TYPE
fi
CC=$CC CXX=$CXX cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE $CXXFLAGS .. && make 
cd ..

if [ $BUILD_OWL -eq 1 ]; then
  python setup.py build_ext --inplace --force $CXXFLAGS
fi
