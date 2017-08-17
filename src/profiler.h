#ifndef _PROFILING_H_
#define _PROFILING_H_

#include <vector>
#include <string>
#include <functional>

namespace thundercat {
  struct TimingInfo {
    long long duration;
    int level;
    std::string description;
    int iterations;
  };
  
  class Profiler {
  public:
    static std::vector<TimingInfo> timingInfos;

    static void recordTime(std::string description, std::function<void()> codeBlock);
    static void recordTime(std::string description, int iterations, std::function<void()> codeBlock);
    static void print();

  private:
    static int timingLevel;
  };
}

#endif
