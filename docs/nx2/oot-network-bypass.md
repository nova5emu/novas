# Ocarina of Time: bypassing the game's anti-aliasing network

The Legend of Zelda: Ocarina of Time (`0400486024DEC000`) runs a tensor network on its tonemapped
image followed by an output pass that sharpens hard. The "Ocarina of Time anti-aliasing" setting
(`oot_anti_aliasing`: Game / No sharpening) can skip both and show the tonemapped image instead.

## What the game does

Found with sampled frame dumps (`capture_frame` with `sampled`), a trace of the texture handles in
the compute passes' constant buffer 0, and pixel statistics, in Kokiri Forest, docked, at 1x with
the built-in Max Resolution patch.

- The 3D renders at the output size (2560x1440 docked), with no jitter and no motion vectors.
- The tonemap pass (fragment shader `bded5ce917183214` outdoors, `3ac011bd2d772994` outdoors at
  another time of day, `6cb06edf14be1d79` indoors) renders a B10G11R11 image.
- The game runs a prepass (`8475d671e6967428`), the network (`42593194bb0b2b32`) and an
  output pass that writes the image the rest of the frame uses, selected by cb0+0xd0.
- The original docked capture identified output key `da6df3e54c716df2`. The S26 handheld
  trace identifies output variants `6f9367bd41dc0b58` (translated shader `e40f5efb5480e692`)
  and `d8efd9f0c95abd80` (`03bf0ffc7d2d0bed`). Both select the final 1920x1080 B10G11R11
  image, texture format `0x21`; the network selects a 1440x810 R32 image, format `0x0f`.
  The handheld output variants cost about 235-290 ms per frame on the S26 in 0.1.2.24.
- The original PC capture's `d9ee412f00ce399a` is a translated shader hash. Earlier documentation
  misclassified it as a native key and classified the handheld output as an unused intermediate.
- The network keeps no history: it is a single-frame spatial smoother (mean |Laplacian| of green
  0.0042 -> 0.0037). The output pass then sharpens hard: 2.2x the input's edge contrast, with
  overshoot up to 1.72 on bright edges. That is the halo users see.
- Only the HUD and a plain copy for presentation follow the output pass.

## What No sharpening does

`ShaderCache::GetComputePipeline` asks `KeplerCompute::BypassOotNetwork` about every native compute
dispatch (`DispatchReplaced` skips it):

- the prepass and the network are skipped while `output_replaced` is true;
- the output pass is replaced by an `AccelerateSurfaceCopy` of the tonemapped image into the output
  pass's own target. Equal dimensions use point filtering. Different dimensions use bilinear
  scaling with independent source and destination rectangles.

The images are learned at run time, not hard-coded: the tonemapped image is render target 0 of the
latest draw whose pipeline has a tonemap fragment shader
(`RasterizerVulkan::TrackOotTonemappedImage`), and the output is whatever the output pass's handle
selects. The source descriptor retains RT0's own dimensions and block-linear tiling, rather than
borrowing the output layout. It is recorded after the tonemap pipeline configures successfully.
Nothing is replaced until a valid tonemapped source has been seen. The prepass and network
are skipped only while the output pass is being replaced: if the output descriptor is not the
expected B10G11R11 block-linear image or the copy fails, the next frame runs the whole network
natively again. The state lives in the rasterizer (`OotNetwork::BypassState`) because the tonemap
pass draws on the 3D engine and the network's passes do not all run on the same compute engine.

The copy accepts single-layer, single-sample B10G11R11 block-linear surfaces, at most 8192 pixels
per dimension, with supported 2D tiling. Source aspect may differ by at most 2% relative to
destination aspect. Scale is limited to 0.5x through 4x per axis, accepting dynamic-resolution
864x480 to 1920x1080. Invalid descriptors, unavailable source
snapshots and failed accelerated copies clear `output_replaced`. Source metadata uses `try_lock`
and is copied before entering the accelerated blit. The blit has `must_accelerate=true`.

`OOT output identity` reports the native output key, replacement result, actual source and
destination sizes and the selected filter. It reports each output variant once for success and
once for failure. The source dimensions are measured from RT0 at runtime; the 1440x810
network descriptor alone does not establish the tonemap dimensions.

The captured `e40f5efb5480e692` SPIR-V samples `cs_tex0_c0` using coordinate transforms from
cb0. For example `%6058` / `%6060` sample at coordinates with components
`(coordinate + 0.5) * cb0[0x78]` and `(coordinate + 0.5) * cb0[0x7c]`. Other coordinate
transforms read float constants at `0x30`, `0x34` and `0x60` through `0x6c`. Their runtime
values are absent from the captured log, and the `03bf0ffc7d2d0bed` SPIR-V is unavailable.
Full-RT versus subrectangle sampling therefore remains unverified. The current replacement
uses the full RT, extending the existing copy policy to the requested dynamic-resolution range.
`OOT output identity` now includes the cb0+0xc0 input descriptor and those float constants,
read once per reported output result, to resolve that question with the next measurement.
The tonemap target log also includes viewport scale/translation and scissor bounds when its
address changes, to check whether the draw fills the descriptor's complete extent.
Descriptor dimensions alone do not prove that 480 is a padded representation of 486.

Replacing a compute dispatch throws `FireEmblemUpscale::DispatchReplaced`, which
`DispatchNativeCompute` catches and reports as a successful dispatch. The enclosing
`NativeQmdScheduler::Signal` then processes all three QMD release slots at words 24, 28 and 52,
followed by dependent-QMD handling. Fire Emblem uses the same path. Release processing already
occurs for replaced dispatches.

With GPU profiling or frame tracing active, `OOT QMD releases` logs the first replaced QMD for
each bypassed native key, after processing the release callbacks. It includes all four raw words
of each slot, the dependent-QMD words and a per-slot result. `queued` records backend acceptance;
`disabled`, `unsupported`, `unmapped` and `not_reached` distinguish the other outcomes. GPU
completion and the guest semaphore write remain separate events.

The available `e40f5efb5480e692` SPIR-V has 26 function stores, 38 workgroup stores and 16
image writes, all targeting `cs_img0_d0`. It has no storage-buffer stores, atomics or memory-copy
instructions. The prepass, network and alternate output dumps are unavailable, so their global
write effects and any guest consumers remain unverified.

Both S26 0.1.2.25 logs contain watchdog reports after the freeze. Without Max Resolution,
the first report at 76.955 s shows the GPU in `gpu_callback` and core 3 in `publish` for about
5.9 seconds. With Max Resolution, the report at 77.789 s shows `gpu_callback` and core 3 in
`decode` for about 7.86 seconds. Both have `query_active=0`. The markers locate a stalled host
callback; thread stacks are still needed to identify its exact wait and establish a lock cycle.

The game's constants (shader hashes, program keys, the cb0 offset) are in
`src/video_core/oot_network.h`.

## Results (Ryzen 7 9800X3D, RTX 5090, Kokiri Forest, 1x)

| Mode | fps | Image |
|---|---|---|
| Game | ~41 | sharpened, halos |
| No sharpening | ~50 | clean; edges aliased as rendered, pairs with resolution scaling |

## Caveats

- Hashes, keys and the offset are specific to the game's main with build id
  `E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348`; a game update needs them re-checked. With other
  keys the bypass does nothing.
- B10G11R11 has little precision in dark tones: without the network's smoothing some banding
  shows in deep shadows.
- Android 0.1.2.23 and 0.1.2.24 never matched the docked output key and retained
  `output_replaced=false`. The handheld output keys now select the copy replacement.
  With `EDEN_NX2_GPU_PROFILE` or `EDEN_DKB_FRAME_TRACE` active, `OOT compute identity` reports
  every new `(native_key, shader_hash)` pair, including non-native mode and shaders without
  lowered native metadata. It includes grid/block size, the raw cb0+0xd0 handle when mapped,
  its image descriptor and bypass state. There is no identity limit. The cache-local registry
  uses `try_lock`; contended attempts are counted in `dropped_identity` on the next new pair.
  A missing native key is represented by zero with `lowered_native=false`.
- Vulkan only: other backends never learn the tonemapped image, so the setting has no effect there.
