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

struct RayTracingApp {
    top_as_memory: vk::DeviceMemory,
    top_as: vk::AccelerationStructureNV,
    bottom_as_memory: vk::DeviceMemory,
    bottom_as: vk::AccelerationStructureNV,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader_binding_table: BufferResource,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    base: Rc<ExampleBase>,
    ray_tracing: Rc<nv::RayTracing>,
    properties: vk::PhysicalDeviceRayTracingPropertiesNV,
}

impl RayTracingApp {
    fn initialize(&mut self) {
        self.create_acceleration_structures();
        self.create_pipeline();
        self.create_shader_binding_table();
        self.create_descriptor_set();
    }

    fn create_acceleration_structures(&mut self) {}

    fn create_pipeline(&mut self) {}

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
        self.shader_binding_table = BufferResource::new(
            table_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            self.base.clone(),
        );
        self.shader_binding_table.store(&table_data);
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

            let image_info = vk::DescriptorImageInfo::builder().build(); //image_view(self.offscreen_image_view).image_layout(vk::ImageLayout::GENERAL).build(); // TODO
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

    fn record_command_buffer(&mut self, command_buffer: vk::CommandBuffer) {
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
                self.shader_binding_table.buffer,
                0,
                self.shader_binding_table.buffer,
                2 * handle_size,
                handle_size,
                self.shader_binding_table.buffer,
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

fn main() {
    unsafe {
        let base = ExampleBase::new(1920, 1080, true);
        let props_rt = nv::RayTracing::get_properties(&base.instance, base.pdevice);
        let ray_tracing = nv::RayTracing::new(&base.instance, &base.device);

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

        return;

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
                |device, draw_command_buffer| {
                    //
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
        /*for pipeline in graphics_pipelines {
            base.device.destroy_pipeline(pipeline, None);
        }
        base.device.destroy_pipeline_layout(pipeline_layout, None);
        base.device
            .destroy_shader_module(vertex_shader_module, None);
        base.device
            .destroy_shader_module(fragment_shader_module, None);
        base.device.free_memory(index_buffer_memory, None);
        base.device.destroy_buffer(index_buffer, None);
        base.device.free_memory(vertex_input_buffer_memory, None);
        base.device.destroy_buffer(vertex_input_buffer, None);
        for framebuffer in framebuffers {
            base.device.destroy_framebuffer(framebuffer, None);
        }
        base.device.destroy_render_pass(renderpass, None);*/
    }
}
