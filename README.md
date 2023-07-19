# rust-wgpu-raytracing
Prototype for a ray-tracer implementation with WebGPU in web browsers, using Rust and WGPU

A browser with WebGPU support is needed (e.g. latest version of Chrome).

To build the WASM modules:

```sh
RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web
```

Then run a web server using the root of this repository as work directory and visit `/web/` path.