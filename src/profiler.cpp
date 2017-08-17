#include "profiler.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace thundercat;

int Profiler::timingLevel = 0;
std::vector<TimingInfo> Profiler::timingInfos;

void Profiler::recordTime(std::string description, std::function<void()> codeBlock) {
  recordTime(description, 1, codeBlock);
}

void Profiler::recordTime(std::string description, int iterations, std::function<void()> codeBlock) {
  timingLevel++;
  auto start = std::chrono::high_resolution_clock::now();
  codeBlock();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  TimingInfo info;
  info.level = timingLevel;
  info.duration = duration;
  info.description = description;
  info.iterations = iterations;
  timingInfos.push_back(info);
  timingLevel--;
}

void Profiler::print() {
  for (auto &info : timingInfos) {
    std::cout << info.level << " ";
    std::cout << std::setw(10) << (info.duration / (double)info.iterations);
    std::cout << " usec.    " << info.description << "\n";
  }
}
