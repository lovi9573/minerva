#include "./minerva_utils.h"
#include <memory>
#include <cstring>
#include <iostream>


namespace libowl {

uint64_t CreateCpuDevice() {
  auto&& ms = minerva::IMinervaSystem::Interface();
  return ms.CreateCpuDevice();
}

uint64_t CreateGpuDevice(int id) {
  auto&& ms = minerva::IMinervaSystem::Interface();
  return ms.CreateGpuDevice(id);
}

uint64_t CreateFpgaDevice(int id) {
  auto&& ms = minerva::IMinervaSystem::Interface();
  return ms.CreateFpgaDevice(id);
}

uint64_t CreateMpiDevice(int rank, int id) {
  auto&& ms = minerva::IMinervaSystem::Interface();
  return ms.CreateMpiDevice(rank, id);
}

int GetGpuDeviceCount() {
  auto&& ms = minerva::IMinervaSystem::Interface();
  return ms.device_manager().GetGpuDeviceCount();
}

int GetMpiNodeCount(){
	auto&& ms = minerva::IMinervaSystem::Interface();
	return ms.device_manager().GetMpiNodeCount();
}

int GetMpiDeviceCount(int rank){
	auto&& ms = minerva::IMinervaSystem::Interface();
	return ms.device_manager().GetMpiDeviceCount(rank);
}

int rank(){
	auto&& ms = minerva::IMinervaSystem::Interface();
	return ms.rank();
}

void WorkerRun(){
	auto&& ms = minerva::IMinervaSystem::Interface();
	ms.WorkerRun();
}

void WaitForAll() {
  auto&& ms = minerva::IMinervaSystem::Interface();
  ms.backend().WaitForAll();
}

void SetDevice(uint64_t id) {
  auto&& ms = minerva::IMinervaSystem::Interface();
  ms.SetDevice(id);
}

void Print_Profiler_Results(){
  auto&& ms = minerva::IMinervaSystem::Interface();
  ms.PrintProfilerResults();
}

minerva::Scale ToScale(std::vector<int>* v) {
  minerva::Scale r(std::move(*v));
  return r;
}

std::vector<int> OfScale(minerva::Scale const& s) {
  std::vector<int> ret;
  for (auto i : s) {
    ret.push_back(i);
  }
  return ret;
}

minerva::NArray FromNumpy(float const* data, minerva::Scale const& scale) {
  auto size = scale.Prod();
  std::shared_ptr<element_t> ptr(new element_t[size], [](element_t* p) {
    delete[] p;
  });
  for(int i = 0; i < size; i++){
	  ptr.get()[i] = data[i];
  }
  //memcpy(ptr.get(), data, size * sizeof(float));
  auto a = minerva::NArray::MakeNArray(scale, ptr);
  minerva::FileFormat fmt;
  fmt.binary = false;
  //a.ToStream(std::cout,fmt);
  return a;
}

void ToNumpy(float* dst, minerva::NArray const& n) {
	//printf("Entering minerva_utils ToNumpy\n");
  auto size = n.Size().Prod();
  auto ptr = n.Get();
  //printf("Minerva Utils ToNumpy recieved %d floats \n", size);
  for (int i = 0; i < size; i++){
	  dst[i] = (float)(ptr.get()[i]);
	  //printf("%f , ",*(ptr.get()+i));
  }
  //memcpy(dst, ptr.get(), size * sizeof(float));
}

}  // namespace libowl

