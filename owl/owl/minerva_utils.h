#pragma once
#include "minerva.h"
#include <vector>
#include <memory>


namespace libowl {

uint64_t CreateCpuDevice();
uint64_t CreateGpuDevice(int);
uint64_t CreateFpgaDevice(int);
uint64_t CreateMpiDevice(int , int );
int GetGpuDeviceCount();
int GetMpiNodeCount();
int GetMpiDeviceCount(int rank);
int rank();
void WorkerRun();
void WaitForAll();
void SetDevice(uint64_t);
void Print_Profiler_Results();
minerva::Scale ToScale(std::vector<int>*);
std::vector<int> OfScale(minerva::Scale const&);

// Support enum class comparison for Cython
template<typename T>
int OfEvilEnumClass(T a) {
  return static_cast<int>(a);
}

template<typename T>
T ToEvilEnumClass(int a) {
  return static_cast<T>(a);
}

minerva::NArray FromNumpy(float const*, minerva::Scale const&);
void ToNumpy(float*, minerva::NArray const&);

}  // namespace libowl


