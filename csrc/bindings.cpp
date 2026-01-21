#include <torch/extension.h>

torch::Tensor naive_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor> flash_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attention", &naive_attention, "V1: Naive 3-pass attention");
    m.def("flash_forward",   &flash_forward,   "V2: Flash Attention 2 forward");
}
