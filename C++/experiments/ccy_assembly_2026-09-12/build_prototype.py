"""Build isolated timing prototypes; production headers and binary are unchanged."""
from pathlib import Path
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
CPP = HERE.parents[1]


def main():
    source = CPP / 'include/ramond/ccy.hpp'
    original = source.read_text()
    code = original

    def replace(old, new):
        nonlocal code
        assert code.count(old) == 1, old[:100]
        code = code.replace(old, new)

    replace('#include <map>', '#include <map>\n#include <optional>')
    replace('    S weight(int edge, int shift) const {', r'''
#ifdef CCY_DENSE_REUSE
    std::array<std::vector<int>, 5> pair_offsets_;
    std::array<int, 5> pair_counts_{};
    std::array<std::vector<std::optional<S>>, 3> dense_vertices_;
    void prepare_dense_vertices(const Index &maximum) {
        for (int edge = 0; edge < 5; edge++) {
            const int cap = edge < 4 ? maximum[edge] : 0;
            pair_offsets_[edge].assign(cap + 1, -1);
            int count = 0;
            for (int shift = 0; shift <= cap; shift++) {
                if (shift == 1) continue;
                pair_offsets_[edge][shift] = count;
                count += cap - shift + 1;
            }
            pair_counts_[edge] = count;
        }
        for (int vertex = 0; vertex < 3; vertex++) {
            const auto edges = vertex < 2 ? std::array<int, 3>{0, 1 + vertex, 3}
                                         : std::array<int, 3>{2, 4, 1};
            dense_vertices_[vertex].resize(size_t(pair_counts_[edges[0]]) *
                                             pair_counts_[edges[1]] * pair_counts_[edges[2]]);
        }
    }
#endif
    S weight(int edge, int shift) const {''')
    replace('''        const std::array<int, 6> key{shifts[edges[0]], shifts[edges[1]], shifts[edges[2]], i, j, k};''', r'''
#ifdef CCY_DENSE_REUSE
        const int a = pair_offsets_[edges[0]][shifts[edges[0]]] + i;
        const int b = pair_offsets_[edges[1]][shifts[edges[1]]] + j;
        const int c = pair_offsets_[edges[2]][shifts[edges[2]]] + k;
        auto &value = dense_vertices_[vertex][(size_t(a) * pair_counts_[edges[1]] + b) *
                                                pair_counts_[edges[2]] + c];
        if (value) {
            vertex_factor_hits++;
            return *value;
        }
        vertex_factor_evaluations++;
        value.emplace(rho(vertex, i, j, k, shifts, vertex != 1));
        return *value;
#else
        const std::array<int, 6> key{shifts[edges[0]], shifts[edges[1]], shifts[edges[2]], i, j, k};''')
    replace('''        return cache.emplace(key, rho(vertex, i, j, k, shifts, vertex != 1)).first->second;
    }''', '''        return cache.emplace(key, rho(vertex, i, j, k, shifts, vertex != 1)).first->second;
#endif
    }''')
    replace('''    std::size_t transitions = 0, seed_terms = 0,''', '''    double preparation_seconds = 0, propagation_seconds = 0, assembly_seconds = 0;
    std::size_t transitions = 0, seed_terms = 0,''')
    replace('''    Series<S> reduced(std::vector<Index> indices) {
        std::sort''', '''    Series<S> reduced(std::vector<Index> indices) {
        double timer = seconds();
        std::sort''')
    replace('''        std::unordered_map<Index, std::map<int, S>, Hash> amplitudes;''', '''#ifdef CCY_DENSE_REUSE
        if (dim_ == 4) prepare_dense_vertices(maximum);
        S transition_scratch;
#endif
        preparation_seconds = seconds() - timer;
        timer = seconds();
        std::unordered_map<Index, std::map<int, S>, Hash> amplitudes;''')
    replace('''                            value += amplitude * inverse_denominator(id, p.id);''', '''#ifdef CCY_DENSE_REUSE
                            if constexpr (std::is_same_v<S, MP>) {
                                const S &inverse = inverse_denominator(id, p.id);
                                mpc_mul(transition_scratch.data(), amplitude.data(), inverse.data(), MPC_RNDNN);
                                mpc_add(value.data(), value.data(), transition_scratch.data(), MPC_RNDNN);
                            } else value += amplitude * inverse_denominator(id, p.id);
#else
                            value += amplitude * inverse_denominator(id, p.id);
#endif''')
    replace('''        std::vector<Index> shifts;''', '''        propagation_seconds = seconds() - timer;
        timer = seconds();
        std::vector<Index> shifts;''')
    replace('''        S scratch;
        for (auto n : indices) {''', r'''
#ifdef CCY_GROUPED_ASSEMBLY
        const size_t box_size = size_t(maximum[0] + 1) * (maximum[1] + 1) *
                                 (maximum[2] + 1) * (maximum[3] + 1);
        require(dim_ == 4 && allowed.size() == box_size, "grouped prototype requires a full four-edge box");
        const size_t stride_d = 1, stride_r = maximum[3] + 1,
                     stride_l = (maximum[2] + 1) * stride_r,
                     stride_a = (maximum[1] + 1) * stride_l;
        std::vector<S *> output(box_size);
        for (auto n : indices)
            output[n[0] * stride_a + n[1] * stride_l + n[2] * stride_r + n[3]] = &answer[n];
        S weighted_middle, scratch;
        for (auto shift : shifts) {
            const S &amplitude = totals.at(shift);
            const std::array<int, 5> s{shift[0], shift[1], shift[2], shift[3], 0};
            for (int l = 0; l <= maximum[1] - shift[1]; l++)
                for (int r = 0; r <= maximum[2] - shift[2]; r++) {
                    const S &middle = vertex_factor(2, r, 0, l, s);
                    if constexpr (std::is_same_v<S, MP>)
                        mpc_mul(weighted_middle.data(), amplitude.data(), middle.data(), MPC_RNDNN);
                    else weighted_middle = amplitude * middle;
                    const size_t base = shift[0] * stride_a + (shift[1] + l) * stride_l +
                                          (shift[2] + r) * stride_r + shift[3] * stride_d;
                    for (int a = 0; a <= maximum[0] - shift[0]; a++)
                        for (int d = 0; d <= maximum[3] - shift[3]; d++) {
                            const S &left = vertex_factor(0, a, l, d, s),
                                    &right = vertex_factor(1, a, r, d, s);
                            S &value = *output[base + a * stride_a + d];
                            if constexpr (std::is_same_v<S, MP>) {
                                mpc_mul(scratch.data(), left.data(), right.data(), MPC_RNDNN);
                                mpc_mul(scratch.data(), scratch.data(), weighted_middle.data(), MPC_RNDNN);
                                mpc_add(value.data(), value.data(), scratch.data(), MPC_RNDNN);
                            } else value += left * right * weighted_middle;
                            seed_terms++;
                        }
                }
        }
#else
        S scratch;
        for (auto n : indices) {''')
    replace('''        return answer;
    }
};''', '''#endif
        assembly_seconds = seconds() - timer;
        return answer;
    }
};''')
    (HERE / 'ccy_prototype.hpp').write_text(code)
    commands = {}
    for name, flags in [('baseline', []), ('grouped', ['CCY_GROUPED_ASSEMBLY']),
                        ('dense', ['CCY_DENSE_REUSE']),
                        ('combined', ['CCY_GROUPED_ASSEMBLY', 'CCY_DENSE_REUSE'])]:
        commands[name] = ['clang++', '-O3', '-std=c++17', '-Wall', '-Wextra',
                          *['-D' + flag for flag in flags], '-I' + str(CPP / 'include'),
                          '-I' + str(CPP / 'include/ramond'), '-I/opt/homebrew/include',
                          str(HERE / 'driver.cpp'), '-L/opt/homebrew/lib', '-lmpc', '-lmpfr',
                          '-lgmpxx', '-lgmp', '-framework', 'Accelerate', '-o', str(HERE / name)]

    def build(item):
        name, command = item
        result = subprocess.run(command, capture_output=True, text=True)
        (HERE / (name + '_build.log')).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(name + ': ' + result.stderr[-3000:])
        return name

    with ThreadPoolExecutor(max_workers=2) as pool:
        for name in pool.map(build, commands.items()):
            print('Built', name, flush=True)
    record = {'production_ccy_sha256': hashlib.sha256(original.encode()).hexdigest(),
              'prototype_sha256': hashlib.sha256(code.encode()).hexdigest(), 'commands': commands}
    (HERE / 'build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
