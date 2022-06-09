#pragma once
#include <chrono>

class CScopedWindowsPerformanceTimer
{
	std::chrono::time_point<std::chrono::steady_clock> start;
	std::chrono::time_point<std::chrono::steady_clock> stop;
	double& secsTimer;
public:
	CScopedWindowsPerformanceTimer(double& dTimeElapsed);
	~CScopedWindowsPerformanceTimer(void);
};
