#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "custom_gpt/tokenizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(custom_tokenizer, m) {
    m.doc() = "BPE Tokenizer";

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())   // no-arg constructor

        .def("train", &Tokenizer::train,
             py::arg("filepath"),
             py::arg("target_vocab_size"))

        .def("encode", &Tokenizer::encode,
             py::arg("text"))

        .def("decode", &Tokenizer::decode,
             py::arg("ids"),
             py::arg("should_separate") = false)  // preserve the default

        .def("print_vocab", &Tokenizer::print_vocab)

        // vocab_size is private — expose it via a lambda instead
        .def("vocab_size", &Tokenizer::get_vocab_size);
}
