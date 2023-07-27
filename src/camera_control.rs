use winit::event::WindowEvent;
use crate::Camera;

pub trait CameraController {
    fn process_events(&mut self, event: &WindowEvent) -> bool;
    fn update_camera(&self, camera: &mut Camera);
}
