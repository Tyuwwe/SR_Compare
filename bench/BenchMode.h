#pragma once
namespace sr {
// Performance benchmark mode: drives `sr_compare viewer` child processes over
// an algorithm x resolution matrix, aggregates per-frame GPU timings and VRAM
// from their --frame-times CSVs, and writes one summary CSV consumable by
// metrics/report.py.
int runBenchMode(int argc, char** argv);
} // namespace sr
