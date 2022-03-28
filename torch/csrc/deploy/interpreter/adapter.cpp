#include <Python.h>
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <torch/csrc/deploy/Exception.h>
#include <torch/csrc/deploy/interpreter/builtin_registry.h>
#include <torch/csrc/jit/python/pybind_utils.h>

using at::IValue;
using torch::deploy::Obj;

namespace torch {
namespace deploy {

Obj fromIValue(IValue value) override {
    return torch::jit::toPyObject(value);
}

IValue toIValue(Obj obj) const override {
    return torch::jit::toTypeInferredIValue(obj);
}

} // namespace deploy
} // namespace torch
