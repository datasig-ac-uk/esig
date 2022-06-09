#include "stdafx.h"
#include "ScopedWindowsPerformanceTimer.h"
#include <chrono>
#include <iostream>

CScopedWindowsPerformanceTimer::CScopedWindowsPerformanceTimer(double& dTimeElapsed)
	: start{ std::chrono::steady_clock::now() }, secsTimer(dTimeElapsed)
{}


CScopedWindowsPerformanceTimer::~CScopedWindowsPerformanceTimer(void)
{
	stop = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	secsTimer = time_span.count();
}