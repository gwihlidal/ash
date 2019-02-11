export SDK_PATH=/Users/graham/src/github.com/gwihlidal/render-hal-rs/render-hal-vk/redist/vulkansdk-macos-1.1.85.0
export DYLD_LIBRARY_PATH=$SDK_PATH/macOS/lib
export VK_ICD_FILENAMES=$SDK_PATH/macOS/etc/vulkan/icd.d/MoltenVK_icd.json
export VK_LAYER_PATH=$SDK_PATH/macOS/etc/vulkan/explicit_layer.d
export VK_LOADER_DEBUG=all
export RUST_BACKTRACE=1

cargo run --bin nv_ray_tracing