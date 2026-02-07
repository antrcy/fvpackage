#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

class Timer {
    using double_second_time = std::chrono::duration<double, std::ratio<1, 1>>;

    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    std::chrono::nanoseconds cumulateTime;

public:
    // CONSTRUCTOR
    Timer() : cumulateTime(std::chrono::nanoseconds::zero()) { start(); }

    Timer(const Timer& other) = default;
    Timer& operator=(const Timer& other) = default;
    Timer(Timer&& other) = delete;
    Timer& operator=(Timer&& other) = delete;

    void reset() {
        startTime = std::chrono::high_resolution_clock::time_point();
        endTime = std::chrono::high_resolution_clock::time_point();
        cumulateTime = std::chrono::nanoseconds::zero();
        start();
    }

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        endTime = std::chrono::high_resolution_clock::now();
        cumulateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
    }

    double getElapsed() const {
        return std::chrono::duration_cast<double_second_time>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime)).count();
    }

    double getCumulated() const {
        return std::chrono::duration_cast<double_second_time>(cumulateTime).count();
    }

    double stopAndGetElapsed() {
        stop();
        return getElapsed();
    }
};

#endif