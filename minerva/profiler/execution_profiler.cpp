#include "profiler/execution_profiler.h"
#include <cstdio>
#include "common/element.h"

using namespace std;

namespace minerva {

ExecutionProfiler::ExecutionProfiler() {
}

ExecutionProfiler::~ExecutionProfiler() {
  for (auto it : time_) {
    delete[] (it.second);
  }
}

void ExecutionProfiler::RecordTime(TimerType type, const string& name, const Timer& timer) {
  lock_guard<mutex> lock_(m_);
  auto it = time_.find(name);
  if (it == time_.end()) {  // Not existent before
    it = time_.insert({name, new double[static_cast<size_t>(TimerType::kEnd)]()}).first;
  }
  (it->second)[static_cast<size_t>(type)] += timer.ReadMicrosecond();
  (it->second)[static_cast<size_t>(TimerType::kCount)] += .5;
}

void ExecutionProfiler::Reset() {
  lock_guard<mutex> lock_(m_);
  for (auto it : time_) {
    delete[] (it.second);
  }
  time_.clear();
}

void ExecutionProfiler::PrintResult() {
  printf("%33s|%6sMemory%8sCalculation%8sCount\n", "", "", "", "");
  for (int i = 0; i < 33; ++i) {
    printf("-");
  }
  printf("|");
  for (int i = 0; i < 34; ++i) {
    printf("-");
  }
  printf("\n");
  float allmemtime = 0;
  float allcaltime = 0;
  for (auto it : time_) {
    printf("%32.32s | %16f %16f %16d\n", it.first.c_str(), it.second[0], it.second[1], static_cast<int>(it.second[2]));
	allmemtime += it.second[0];
	allcaltime += it.second[1];
  }
  printf("All Mem Time: %16f  All Cal Time: % 16f  All Time: %16f\n", allmemtime, allcaltime ,allmemtime + allcaltime);
#ifdef FIXED_POINT
  FixedPoint<FIXED_POINT_DOUBLE_WIDE_TYPE,FIXED_POINT_TYPE,FIXED_POINT_FRACTION_WIDTH_N>::Report();
#endif
}

}  // namespace minerva
