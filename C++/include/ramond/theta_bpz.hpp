#pragma once
#include "number.hpp"

namespace ramond {
// BPZ sewing in the paper's theta-channel edge order: an odd tube gives
// -i, and the two infinity-slot vertices give -1 when edge 1 is odd.
template <class S> S theta_bpz_weight(int parity_mask) {
    S weight(1), minus_i(Machine(0, -1));
    for (int edge = 0; edge < 3; edge++)
        if (parity_mask & (1 << edge)) weight *= minus_i;
    if (parity_mask & 1) weight = -weight;
    return weight;
}
template <class S, class Graph> S plumbing_bpz_weight(const Graph &graph, int parity_mask) {
    S weight(1), minus_i(Machine(0, -1)), i(Machine(0, 1));
    for (int edge = 0; edge < graph.edges; edge++)
        if (parity_mask & (1 << edge)) weight *= minus_i;
    for (const auto &slots : graph.slots)
        if (parity_mask & (1 << slots[0])) weight *= i;
    return weight;
}
} // namespace ramond
