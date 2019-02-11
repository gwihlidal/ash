extern crate ash;
extern crate examples;

use ash::extensions::nv;
use ash::util::*;
use ash::version::InstanceV1_1;
use ash::vk;
use examples::*;
use std::default::Default;
use std::ffi::CString;
use std::fs::File;
use std::mem;
use std::mem::align_of;
use std::path::Path;
use std::rc::Rc;

#[derive(Clone)]
struct BufferResource {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    base: Rc<ExampleBase>,
}

impl BufferResource {
    fn new(
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        base: Rc<ExampleBase>,
    ) -> Self {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            let buffer = base.device.create_buffer(&buffer_info, None).unwrap();

            let memory_req = base.device.get_buffer_memory_requirements(buffer);

            let memory_index = find_memorytype_index(
                &memory_req,
                &base.device_memory_properties,
                memory_properties,
            )
            .unwrap();

            let allocate_info = vk::MemoryAllocateInfo {
                allocation_size: memory_req.size,
                memory_type_index: memory_index,
                ..Default::default()
            };

            let memory = base.device.allocate_memory(&allocate_info, None).unwrap();

            base.device.bind_buffer_memory(buffer, memory, 0).unwrap();

            BufferResource {
                buffer,
                memory,
                size,
                base,
            }
        }
    }

    fn store<T: Copy>(&mut self, data: &[T]) {
        unsafe {
            let size = (std::mem::size_of::<T>() * data.len()) as u64;
            let mapped_ptr = self.map(size);
            let mut mapped_slice = Align::new(mapped_ptr, align_of::<T>() as u64, size);
            mapped_slice.copy_from_slice(&data);
            self.unmap();
        }
    }

    fn map(&mut self, size: vk::DeviceSize) -> *mut std::ffi::c_void {
        unsafe {
            let data: *mut std::ffi::c_void = self
                .base
                .device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            data
        }
    }

    fn unmap(&mut self) {
        unsafe {
            self.base.device.unmap_memory(self.memory);
        }
    }
}

impl Drop for BufferResource {
    fn drop(&mut self) {
        unsafe {
            self.base.device.destroy_buffer(self.buffer, None);
            self.base.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone)]
struct RayTracingApp {
    base: Rc<ExampleBase>,
    ray_tracing: Rc<nv::RayTracing>,
    properties: vk::PhysicalDeviceRayTracingPropertiesNV,
    top_as_memory: vk::DeviceMemory,
    top_as: vk::AccelerationStructureNV,
    bottom_as_memory: vk::DeviceMemory,
    bottom_as: vk::AccelerationStructureNV,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader_binding_table: Option<BufferResource>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl RayTracingApp {
    fn new(
        base: Rc<ExampleBase>,
        ray_tracing: Rc<nv::RayTracing>,
        properties: vk::PhysicalDeviceRayTracingPropertiesNV,
    ) -> Self {
        RayTracingApp {
            base,
            ray_tracing,
            properties,
            top_as_memory: vk::DeviceMemory::null(),
            top_as: vk::AccelerationStructureNV::null(),
            bottom_as_memory: vk::DeviceMemory::null(),
            bottom_as: vk::AccelerationStructureNV::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            shader_binding_table: None,
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
        }
    }

    fn initialize(&mut self) {
        self.create_acceleration_structures();
        self.create_pipeline();
        self.create_shader_binding_table();
        self.create_descriptor_set();
    }

    fn release(&mut self) {
        unsafe {
            self.base.device.device_wait_idle().unwrap();

            self.ray_tracing
                .destroy_acceleration_structure(self.top_as, None);
            self.base.device.free_memory(self.top_as_memory, None);

            self.ray_tracing
                .destroy_acceleration_structure(self.bottom_as, None);
            self.base.device.free_memory(self.bottom_as_memory, None);

            self.base
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.shader_binding_table = None;

            self.base.device.destroy_pipeline(self.pipeline, None);
            self.base
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.base
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }

    fn create_acceleration_structures(&mut self) {}

    fn create_pipeline(&mut self) {
        unsafe {
            let accel_binding = vk::DescriptorSetLayoutBinding::builder()
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_NV)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_NV)
                .build();

            let output_binding = vk::DescriptorSetLayoutBinding::builder()
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_NV)
                .build();

            let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&[accel_binding, output_binding])
                .build();
            self.base
                .device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap();

            //
            let mut rgen_spv_file =
                File::open(Path::new("shader/nv_ray_tracing/triangle.rgen.spv"))
                    .expect("Could not find triangle.rgen.spv.");
            let mut chit_spv_file =
                File::open(Path::new("shader/nv_ray_tracing/triangle.chit.spv"))
                    .expect("Could not find triangle.chit.spv.");
            let mut miss_spv_file =
                File::open(Path::new("shader/nv_ray_tracing/triangle.miss.spv"))
                    .expect("Could not find triangle.miss.spv.");

            let rgen_code = read_spv(&mut rgen_spv_file).expect("Failed to read raygen spv file");
            let rgen_shader_info = vk::ShaderModuleCreateInfo::builder().code(&rgen_code);
            let rgen_shader_module = self
                .base
                .device
                .create_shader_module(&rgen_shader_info, None)
                .expect("Raygen shader module error");

            let chit_code = read_spv(&mut chit_spv_file).expect("Failed to read chit spv file");
            let chit_shader_info = vk::ShaderModuleCreateInfo::builder().code(&chit_code);
            let chit_shader_module = self
                .base
                .device
                .create_shader_module(&chit_shader_info, None)
                .expect("Closest-hit shader module error");

            let miss_code = read_spv(&mut miss_spv_file).expect("Failed to read miss spv file");
            let miss_shader_info = vk::ShaderModuleCreateInfo::builder().code(&miss_code);
            let miss_shader_module = self
                .base
                .device
                .create_shader_module(&miss_shader_info, None)
                .expect("Miss shader module error");

            let layouts = vec![self.descriptor_set_layout];
            let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);

            self.pipeline_layout = self
                .base
                .device
                .create_pipeline_layout(&layout_create_info, None)
                .unwrap();

            let shader_groups = vec![
                // group0 = [ raygen ]
                vk::RayTracingShaderGroupCreateInfoNV::builder()
                    .ty(vk::RayTracingShaderGroupTypeNV::GENERAL)
                    .general_shader(0)
                    .build(),
                // group1 = [ chit ]
                vk::RayTracingShaderGroupCreateInfoNV::builder()
                    .ty(vk::RayTracingShaderGroupTypeNV::TRIANGLES_HIT_GROUP)
                    .closest_hit_shader(1)
                    .build(),
                // group2 = [ miss ]
                vk::RayTracingShaderGroupCreateInfoNV::builder()
                    .ty(vk::RayTracingShaderGroupTypeNV::GENERAL)
                    .general_shader(2)
                    .build(),
            ];

            let shader_stages = vec![
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::RAYGEN_NV)
                    .module(rgen_shader_module)
                    .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::CLOSEST_HIT_NV)
                    .module(chit_shader_module)
                    .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::MISS_NV)
                    .module(miss_shader_module)
                    .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build(),
            ];

            let pipeline_info = vk::RayTracingPipelineCreateInfoNV::builder()
                .stages(&shader_stages)
                .groups(&shader_groups)
                .max_recursion_depth(1)
                .layout(self.pipeline_layout)
                .build();

            self.pipeline = self
                .ray_tracing
                .create_ray_tracing_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0];
        }
    }

    fn create_shader_binding_table(&mut self) {
        let group_count = 3; // Listed in vk::RayTracingPipelineCreateInfoNV
        let table_size = (self.properties.shader_group_handle_size * group_count) as u64;
        let mut table_data: Vec<u8> = vec![0u8; table_size as usize];
        unsafe {
            self.ray_tracing
                .get_ray_tracing_shader_group_handles(
                    self.pipeline,
                    0,
                    group_count,
                    &mut table_data,
                )
                .unwrap();
        }
        let mut shader_binding_table = BufferResource::new(
            table_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            self.base.clone(),
        );
        shader_binding_table.store(&table_data);
        self.shader_binding_table = Some(shader_binding_table);
    }

    fn create_descriptor_set(&mut self) {
        unsafe {
            let descriptor_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
                    descriptor_count: 1,
                },
            ];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&descriptor_sizes)
                .max_sets(1);

            self.descriptor_pool = self
                .base
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap();

            let layouts = vec![self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);

            let descriptor_sets = self
                .base
                .device
                .allocate_descriptor_sets(&alloc_info)
                .unwrap();
            self.descriptor_set = descriptor_sets[0];

            let mut accel_info = vk::WriteDescriptorSetAccelerationStructureNV::builder()
                .acceleration_structures(&[self.top_as])
                .build();
            let accel_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_NV)
                .next(&mut accel_info)
                .build();

            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::GENERAL)
                .build(); //image_view(self.offscreen_image_view) // TODO
            let image_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&[image_info])
                .build();

            self.base
                .device
                .update_descriptor_sets(&[accel_write, image_write], &[]);
        }
    }

    fn record_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        if let Some(ref shader_binding_table) = self.shader_binding_table {
            let handle_size = self.properties.shader_group_handle_size as u64;
            // |[ raygen shader ]|[ hit shader  ]|[ miss shader ]|
            // |                 |               |               |
            // | 0               | 1             | 2             | 3
            unsafe {
                self.base.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_NV,
                    self.pipeline,
                );
                self.base.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_NV,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                self.ray_tracing.cmd_trace_rays(
                    command_buffer,
                    shader_binding_table.buffer,
                    0,
                    shader_binding_table.buffer,
                    2 * handle_size,
                    handle_size,
                    shader_binding_table.buffer,
                    1 * handle_size,
                    handle_size,
                    vk::Buffer::null(),
                    0,
                    0,
                    1920,
                    1080,
                    1,
                )
            }
        }
    }
}

fn main() {
    unsafe {
        let base = Rc::new(ExampleBase::new(1920, 1080, true));
        let props_rt = nv::RayTracing::get_properties(&base.instance, base.pdevice);
        let ray_tracing = Rc::new(nv::RayTracing::new(&base.instance, &base.device));
        let mut app = RayTracingApp::new(base.clone(), ray_tracing, props_rt);
        app.initialize();

        println!("NV Ray Tracing Properties:");
        println!(
            " shader_group_handle_size: {}",
            props_rt.shader_group_handle_size
        );
        println!(" max_recursion_depth: {}", props_rt.max_recursion_depth);
        println!(
            " max_shader_group_stride: {}",
            props_rt.max_shader_group_stride
        );
        println!(
            " shader_group_base_alignment: {}",
            props_rt.shader_group_base_alignment
        );
        println!(" max_geometry_count: {}", props_rt.max_geometry_count);
        println!(" max_instance_count: {}", props_rt.max_instance_count);
        println!(" max_triangle_count: {}", props_rt.max_triangle_count);
        println!(
            " max_descriptor_set_acceleration_structures: {}",
            props_rt.max_descriptor_set_acceleration_structures
        );

        base.render_loop(|| {
            let (present_index, _) = base
                .swapchain_loader
                .acquire_next_image(
                    base.swapchain,
                    std::u64::MAX,
                    base.present_complete_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();

            record_submit_commandbuffer(
                &base.device,
                base.draw_command_buffer,
                base.present_queue,
                &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                &[base.present_complete_semaphore],
                &[base.rendering_complete_semaphore],
                |_device, command_buffer| {
                    app.record_command_buffer(command_buffer);
                },
            );

            let wait_semaphors = [base.rendering_complete_semaphore];
            let swapchains = [base.swapchain];
            let image_indices = [present_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&wait_semaphors)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            base.swapchain_loader
                .queue_present(base.present_queue, &present_info)
                .unwrap();
        });

        base.device.device_wait_idle().unwrap();
        app.release();
    }
}
