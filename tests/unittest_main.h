#pragma once
#include <minerva.h>
#include <system/minerva_system.h>
#include <cstdint>
#include <gtest/gtest.h>

extern uint64_t cpu_device;
#ifdef HAS_CUDA
extern uint64_t gpu_device;
#endif

