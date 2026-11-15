# Vulkan hang trace

Launch with `EDEN_NX2_HANG_TRACE=1`. The setting is read once and defaults to off.
The trace does not change submission order, waits, shader rejection, completion
policy or presentation recovery. When enabled on a device advertising
VK_AMD_buffer_marker, it adds a 64 KiB coherent mapped buffer and GPU marker
commands. This can change timing and cost GPU time.

When off, command/queue wrappers take a cached-flag fast path directly to the
original Vulkan call, without building records, reading thread-local trace
context, accessing rings/clocks or constructing marker scopes. Device creation
does not enable VK_AMD_buffer_marker or allocate its buffer. Resources have an
additional opt-in flag; no allocation registry entries or UBO samples are made
when disabled. This is not literally zero instruction overhead.

The first one-second timeline wait timeout writes `HANG` records at Error level.
A second snapshot (`persistent=true`) is allowed once per process when a waiter
observes no timeline progress for at least five seconds. Progress resets that
waiter's timer. Concurrent dump attempts do not block; a busy attempt can retry
on the next timeout. The Android HangGuard retains its separate >=3 s notification.
The diagnostic never waits for the worker or takes its submission mutex.
Ring access uses a single nonblocking atomic-flag attempt. Contention drops,
overwrites, unavailable snapshots, and eviction of the first incomplete tick
are reported rather than interpreted as absence of work.

There are 256 work records per tick slot, 64 tick slots, and 2048 lifecycle
events. Work is captured at the Vulkan command wrappers on the worker, including
internal compute passes and ASTC. These are commands recorded on the host, not
proof that the GPU reached or executed them. The dump lists retained ticks in
`gpu+1 .. gpu+4`, in ascending order. Lifecycle events include completed producers so
binary semaphore ownership can be reconstructed. `seq` correlates both rings;
`ns` is a steady-clock timestamp. Concurrent event insertion can reorder sequence
numbers within the event ring; use `seq` when comparing threads.

Each event labels its data fields. All `data` fields are hexadecimal, including
indices, counts, and VkResult (negative results are converted to u64). Work fields:

1. Command buffer handle.
2. Pipeline handle (process-local; never use this as a shader reject hash).
3. Graphics pipeline cache-key hash, or zero when unavailable.
4. Native guest compute shader hash, or zero for internal shaders/unavailable.
5. Direct x/count, or indirect buffer offset.
6. Direct y/instance count (draw variants label their operation).
7. Direct z/stride.
8. Indirect Vulkan buffer handle, not a guest virtual address.

Compute records include their recording call site and function, identifying
ASTC and other internal passes even though those pipelines have no native guest
shader hash. Hashes print all 16 hex digits. No synthetic hash is substituted for
an unavailable pipeline/shader hash. Internal blits identify source/destination
images in the x/y fields; their graphics binding fields are not applicable.
Other transfer/query/clear/boundary records likewise reuse the work layout for
operation arguments; their cached graphics binding is not evidence that they
execute that shader.

`tick-reserved` and `chunk-enqueue` do not prove submission. `submit-enter` means
the worker entered the submission path; `submit-return result=0` proves it returned
VK_SUCCESS. It does not prove GPU completion. Compare the timeline at the stall
with the submit and presentation events. Presentation records include attempts,
caught errors, acquire entry/result, blit entry/result, discard, frame-fence waits,
and present entry/result. Blits declare two binary waits and one signal; discard
declares one binary wait and no signals. These waits have no timeline value.

The bounded trace may omit older producers or the first incomplete tick after
more than 64 ticks are recorded. Missing/overwritten evidence is inconclusive.
This diagnostic does not query the signaled state of binary semaphores, inspect
GPU instructions, or replace a driver capture. It is intended to choose the next
shader or presentation investigation, not automatically classify a root cause.

## Resource snapshots and GPU progress

See [hang-trace-resources.md](hang-trace-resources.md) for the optional resource
records, AMD markers, command-buffer boundaries and queue events. Sort CPU
records by seq, then interpret GPU evidence in actual submission order. A
matching after token establishes progress through that marker's scope, not
completion of a timeline signal or permission to reuse resources. An open
marker does not uniquely identify the guilty command. Missing records or drops
make negative conclusions inconclusive.

The documented 2026-11-14 Radeon 890M case reached both end markers while the
completion primitive stopped advancing. Disabling HAGS avoided the hang in
two PREFLUSH-enabled runs, versus seven reported failures with HAGS enabled.
Legacy-submit and per-submit-fence experiments also failed and were removed.
The evidence strongly implicates an AMD/HAGS interaction; the separate possible
CPU/writeback race remains unresolved. See the case study for drivers, durations,
local log names, and limits of this conclusion.
