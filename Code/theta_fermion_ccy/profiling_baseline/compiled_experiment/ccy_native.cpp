// CCY recursion and memoization only. Numerical operations remain CPython's
// PyNumber_Multiply/Add, with the same term order as punctured_ccy.py.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include <exception>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct PythonError {};

struct Ref {
    PyObject* p = nullptr;
    Ref() = default;
    explicit Ref(PyObject* value) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    Ref(Ref&& other) noexcept : p(std::exchange(other.p, nullptr)) {}
    Ref& operator=(Ref&& other) noexcept {
        if (this != &other) { Py_XDECREF(p); p = std::exchange(other.p, nullptr); }
        return *this;
    }
    PyObject* release() { return std::exchange(p, nullptr); }
};

Ref checked(PyObject* p) {
    if (!p) throw PythonError{};
    return Ref(p);
}

using Id = uint32_t;
Id as_id(PyObject* p) {
    unsigned long n = PyLong_AsUnsignedLong(p);
    if (PyErr_Occurred()) throw PythonError{};
    if (n > std::numeric_limits<Id>::max()) {
        PyErr_SetString(PyExc_OverflowError, "CCY integer ID exceeds uint32");
        throw PythonError{};
    }
    return static_cast<Id>(n);
}

uint64_t mix(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
struct Key {
    Id c, shift, level;
    bool operator==(const Key& b) const {
        return c == b.c && shift == b.shift && level == b.level;
    }
};
struct Hash {
    size_t operator()(const Key& k) const {
        return mix((uint64_t(k.shift) << 32) | k.level) ^ mix(k.c);
    }
};
struct PairHash {
    size_t operator()(uint64_t k) const { return mix(k); }
};
struct Step { Id edge, r, s, lower, label; bool terminal; };
struct Transition { Id pole = 0, shift = 0; Ref factor; };
struct Counter { size_t hits = 0, misses = 0; };

Ref cache_info(const Counter& c, size_t size) {
    return checked(Py_BuildValue("{s:K,s:K,s:O,s:K}",
        "hits", (unsigned long long)c.hits,
        "misses", (unsigned long long)c.misses,
        "maxsize", Py_None, "currsize", (unsigned long long)size));
}

class Evaluator {
    PyObject *global_fn, *transition_fn, *steps_fn; // borrowed during evaluate()
    std::vector<Ref> integer_objects;
    std::unordered_map<Id, std::vector<Step>> level_steps;
    std::unordered_map<Key, Id, Hash> labels;
    std::unordered_map<Key, Ref, Hash> coefficients;
    std::unordered_map<uint64_t, Ref, PairHash> globals;
    std::unordered_map<Key, Transition, Hash> transitions;
    Counter coefficient_count, global_count, transition_count;

    PyObject* integer(Id id) {
        if (id >= integer_objects.size()) integer_objects.resize(size_t(id) + 1);
        if (!integer_objects[id].p)
            integer_objects[id] = checked(PyLong_FromUnsignedLong(id));
        return integer_objects[id].p;
    }

    const std::vector<Step>& steps(Id level) {
        auto found = level_steps.find(level);
        if (found != level_steps.end()) return found->second;
        PyObject* args[] = {integer(level)};
        Ref raw = checked(PyObject_Vectorcall(steps_fn, args, 1, nullptr));
        Ref seq = checked(PySequence_Fast(raw.p, "CCY steps must be a sequence"));
        std::vector<Step> result;
        result.reserve(PySequence_Fast_GET_SIZE(seq.p));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq.p); ++i) {
            Ref row = checked(PySequence_Fast(PySequence_Fast_GET_ITEM(seq.p, i),
                                             "CCY step must be a sequence"));
            if (PySequence_Fast_GET_SIZE(row.p) != 5) {
                PyErr_SetString(PyExc_ValueError, "CCY step must have five entries");
                throw PythonError{};
            }
            auto v = PySequence_Fast_ITEMS(row.p);
            Id edge = as_id(v[0]), r = as_id(v[1]), s = as_id(v[2]);
            int terminal = PyObject_IsTrue(v[4]);
            if (terminal < 0) throw PythonError{};
            Key label_key{edge, r, s};
            auto [entry, inserted] = labels.emplace(label_key, Id(labels.size()));
            result.push_back({edge, r, s, as_id(v[3]), entry->second, bool(terminal)});
        }
        // unordered_map node references survive later insertions/rehashes.
        return level_steps.emplace(level, std::move(result)).first->second;
    }

    PyObject* global(Id shift, Id level) {
        uint64_t key = (uint64_t(shift) << 32) | level;
        auto found = globals.find(key);
        if (found != globals.end()) { ++global_count.hits; return found->second.p; }
        ++global_count.misses;
        PyObject* args[] = {integer(shift), integer(level)};
        Ref value = checked(PyObject_Vectorcall(global_fn, args, 2, nullptr));
        return globals.emplace(key, std::move(value)).first->second.p;
    }

    const Transition& transition(Id c, Id shift, const Step& step) {
        Key key{c, shift, step.label};
        auto found = transitions.find(key);
        if (found != transitions.end()) { ++transition_count.hits; return found->second; }
        ++transition_count.misses;
        PyObject* args[] = {integer(c), integer(shift), integer(step.edge),
                            integer(step.r), integer(step.s)};
        Ref raw = checked(PyObject_Vectorcall(transition_fn, args, 5, nullptr));
        Transition value;
        if (raw.p != Py_None) {
            if (!PyTuple_Check(raw.p) || PyTuple_GET_SIZE(raw.p) != 3) {
                PyErr_SetString(PyExc_TypeError, "CCY transition must be None or a triple");
                throw PythonError{};
            }
            value.pole = as_id(PyTuple_GET_ITEM(raw.p, 0));
            value.factor = Ref(Py_NewRef(PyTuple_GET_ITEM(raw.p, 1)));
            value.shift = as_id(PyTuple_GET_ITEM(raw.p, 2));
        }
        return transitions.emplace(key, std::move(value)).first->second;
    }

    PyObject* coefficient(Id c, Id shift, Id level) {
        Key key{c, shift, level};
        auto found = coefficients.find(key);
        if (found != coefficients.end()) {
            ++coefficient_count.hits;
            return found->second.p;
        }
        ++coefficient_count.misses;
        if (Py_EnterRecursiveCall(" in compiled CCY recurrence")) throw PythonError{};
        struct Leave { ~Leave() { Py_LeaveRecursiveCall(); } } leave;
        Ref value(Py_NewRef(global(shift, level)));
        for (const auto& step : steps(level)) {
            const auto& tr = transition(c, shift, step);
            if (!tr.factor.p) continue; // exactly the Python None transition
            PyObject* lower = step.terminal ? global(tr.shift, step.lower) :
                coefficient(tr.pole, tr.shift, step.lower);
            Ref product = checked(PyNumber_Multiply(tr.factor.p, lower));
            value = checked(PyNumber_Add(value.p, product.p));
        }
        return coefficients.emplace(key, std::move(value)).first->second.p;
    }

public:
    Evaluator(PyObject* g, PyObject* t, PyObject* s)
        : global_fn(g), transition_fn(t), steps_fn(s) {}

    Ref run(PyObject* requested) {
        Ref seq = checked(PySequence_Fast(requested, "CCY levels must be a sequence"));
        Ref values = checked(PyList_New(PySequence_Fast_GET_SIZE(seq.p)));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq.p); ++i) {
            Id level = as_id(PySequence_Fast_GET_ITEM(seq.p, i));
            PyObject* value = steps(level).empty() ? global(0, level) : coefficient(0, 0, level);
            PyList_SET_ITEM(values.p, i, Py_NewRef(value));
            if (PyErr_CheckSignals()) throw PythonError{};
        }
        Ref c = cache_info(coefficient_count, coefficients.size());
        Ref g = cache_info(global_count, globals.size());
        Ref t = cache_info(transition_count, transitions.size());
        Ref info = checked(Py_BuildValue("{s:O,s:O,s:O,s:K}",
            "coefficient", c.p, "global", g.p, "transition", t.p,
            "compiled_level_lists", (unsigned long long)level_steps.size()));
        return checked(Py_BuildValue("(OO)", values.p, info.p));
    }
};

PyObject* evaluate(PyObject*, PyObject* args) {
    PyObject *g, *t, *s, *levels;
    if (!PyArg_ParseTuple(args, "OOOO", &g, &t, &s, &levels)) return nullptr;
    if (!PyCallable_Check(g) || !PyCallable_Check(t) || !PyCallable_Check(s)) {
        PyErr_SetString(PyExc_TypeError, "CCY callbacks must be callable");
        return nullptr;
    }
    try {
        Evaluator evaluator(g, t, s);
        return evaluator.run(levels).release();
    } catch (const PythonError&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"evaluate", evaluate, METH_VARARGS, "Evaluate ordered CCY levels with native recursion caches."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef definition = {PyModuleDef_HEAD_INIT, "_ccy_native", nullptr, -1, methods};
} // namespace

PyMODINIT_FUNC PyInit__ccy_native() { return PyModule_Create(&definition); }
