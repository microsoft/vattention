#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void reshape_and_cache_flash(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping,
  const std::string& kv_cache_dtype);

  void cache_flat(
  torch::Tensor& key,           
  torch::Tensor& value,         
  torch::Tensor& k_cache,      
  torch::Tensor& v_cache,      
  // torch::Tensor& slot_mapping, 
  const std::string& kv_cache_dtype);

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "swap_blocks",
        &swap_blocks,
        "Swap blocks");
    m.def(
        "copy_blocks",
        &copy_blocks,
        "Copy blocks");
    m.def(
        "reshape_and_cache",
        &reshape_and_cache,
        "Reshape and cache");
    m.def(
        "gather_cached_kv",
        &gather_cached_kv,
        "Gather cached kv");
    m.def(
        "reshape_and_cache_flash",
        &reshape_and_cache_flash,
        "Reshape and cache flash");
    m.def(
        "cache_flat",
        &cache_flat,
        "Cache flat");
  }
