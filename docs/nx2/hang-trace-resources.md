# Vulkan hang resource trace

Enable `EDEN_NX2_HANG_TRACE=1 EDEN_NX2_HANG_TRACE_RESOURCES=1` before starting the
process. Both are off by default. Resource records are CPU-side snapshots, not GPU execution timestamps. They add
no GPU waits or readbacks. On supported devices the base hang trace also enables
the separate AMD marker buffer and GPU writes described below.

On a stall, the existing first-incomplete-tick dump also prints resource records
for the same ticks. Sort all records by `seq`: resource and work rings are printed
separately. A compute configuration's records precede its pipeline bind/dispatch;
match command buffer, shader and sequence, not shader alone. Metadata is copied
before handing the scheduler callback to its worker, so later descriptor-queue
reuse cannot change the snapshot.

Each record names its comma-separated fields; `data` values are hexadecimal:

- `compute-descriptors`: descriptor mode (0 sets, 1 push, 2 descriptor buffer),
  descriptor-ring chunk/offset, global fallback and native-bounds flags, and the
  device's enabled core robust-buffer-access feature.
- `compute-source`: maps descriptor payload entry to Vulkan binding and array
  count, descriptor type, guest constant buffer and pointer offset. Storage
  descriptors include their indirect offset (`ffffffff` means direct).
- `compute-buffer`: actual UBO/SSBO VkBuffer, offset and descriptor range. In mode
  2 it also records the device address used to construct the bounded descriptor;
  this is not evidence of PhysicalStorageBuffer shader instructions. A null
  address in that mode uses the descriptor's actual VK_WHOLE_SIZE range.
- `compute-cb0-guest`: guest CB0 address/size and two 64-bit RAM samples at 0xc0
  and 0x150. Valid-mask bits 0/1 distinguish readable samples from missing data.
  These are unsafe/non-flushing reads during Configure, NOT the uploaded UBO
  contents or the values observed by the GPU. The two reads are not atomic as a
  pair. No sample is taken from a disabled, undersized or unmapped CB0.
- `copy-buffer`: each VkBufferCopy's source/destination buffers, offsets and size.
- `barrier*`: stages, access masks, buffer ranges, image layouts/subresources and
  queue-family indices passed through the PipelineBarrier wrapper (including its
  synchronization2 conversion).

UBO/SSBO descriptors are covered; texture/texel descriptors are not decoded by
`compute-buffer`. Image-copy commands have work markers, but their complete
region payloads are not decoded. Direct calls bypassing wrappers are not covered. Resource records have their own bounded ring (1024 records per
slot, 64 tick slots) so they cannot evict the original work records. Overflow and
contention drops are reported; an incomplete trace is not proof of absence.

For the OoT preflush investigation, retain the failing launch environment and
add the resource flag. Disk shader-cache disabling and shader dumps are not
required for this instrumentation. Compare the bound ranges and guest samples
with a GPU capture before attributing a mismatch to CPU writeback.

## AMD GPU progress markers

With EDEN_NX2_HANG_TRACE=1, devices advertising VK_AMD_buffer_marker now enable it.
No separate marker switch is needed. The startup log must say
"HANG AMD buffer markers enabled"; allocation/entry-point failures and unsupported
extensions are reported and leave the CPU trace usable.

The legacy vkCmdWriteBufferMarkerAMD entry point works with either submission path,
including synchronization2. Each traced Draw*, Dispatch*, BlitImage and CopyBuffer
gets TOP_OF_PIPE before and BOTTOM_OF_PIPE after markers. CopyBuffer has a new
work record named "copy"; its region records remain in the resource trace.
The buffer is 64 KiB, persistently mapped, HOST_VISIBLE | HOST_COHERENT, TRANSFER_DST.
There are 8192 pairs. A unique nonzero 32-bit token maps each pair to the full 64-bit
CPU seq, tick and command buffer. Token wrap disables further markers; it never
aliases an old token. CPU initialization occurs only before any submission.
Slots are reused only after the existing timeline/fence reports the owning tick
complete. Exhaustion or metadata-lock contention drops markers without waiting;
check the cumulative dropped count before interpreting gaps.

OnStall reads mapped words directly, without a Vulkan wait or query-result wait.
Each line reports before/after matches and raw tokens. This is a live diagnostic
snapshot, not a synchronized application readback or an atomic snapshot of all
GPU writes. A subsequent persistent dump can distinguish advancing markers.
The summary gives the maximum retained CPU seq with a matching after token and
the minimum retained seq with before but no after (UINT64_MAX if none).
It is computed on the CPU; there is deliberately no shared GPU "last" word:
upload and main command buffers execute in submission order, which can differ
from CPU seq order. The maximum is not a contiguous completion frontier.

TOP can run ahead, and a driver may emit a marker at a later stage. An open pair
is pending work, not proof that its command caused the hang. A completed BOTTOM
establishes completion through that marker's scope. Interpret the set of pairs
in command-buffer/submission order, alongside barriers and submits. Markers can
change timing and cost GPU time; a run that no longer hangs is inconclusive.

## Allocation and upload snapshots

With EDEN_NX2_HANG_TRACE_RESOURCES=1, compute-allocation reports the buffer handle,
allocation generation, actual VkBuffer size, VkDeviceMemory handle, allocation
offset and size, memory property flags, and the generation seen by the worker.
Generation zero means unknown/unavailable (including try-lock contention), not a
zero-sized Vulkan allocation. Generation mismatch detects destruction/reuse
between configuration and recording. This covers buffers created through the
normal VMA allocator and aligned-buffer path; sparse/raw external buffers are
not represented. It does not prove lifetime after recording.

compute-cb0-upload samples offsets 0xC0 and 0x150 from the actual mapped Upload/Stream
allocation selected by the cb0 descriptor, after UpdateBuffers, before queuing
worker recording. Values and validity bits are captured by value. It also covers
reused stream UBO descriptors. Valid bit 0 is 0xC0, bit 1 is 0x150. Zero validity
means unavailable/out of range, never a fallback to guest RAM. The separate
compute-cb0-guest record remains available for comparison. These are the CPU-side
bytes supplied for upload, not an assertion about bytes subsequently observed
by the GPU, nor a GPU readback of device-local UBOs.

## Command-buffer boundaries and queue coverage

The worker emits main-begin/upload-begin immediately after beginning each buffer
(the first pair is tagged when the worker starts), and upload-end/main-end after
all recorded work and final barriers/timestamps, before vkEndCommandBuffer.
Each boundary has the same TOP/BOTTOM pair as work commands. Match its work seq
to the AMD marker record; the after word of main-end/upload-end is the final
BOTTOM for that command buffer. It does not prove the timeline signal completed.

Added work pairs cover CopyBufferToImage, CopyImage, CopyImageToBuffer,
ResolveImage, ClearColorImage, ClearAttachments, Begin/EndRenderPass,
Begin/End/ResetQueryPool, WriteTimestamp, CopyQueryPoolResults and FillBuffer.
There are no vkCmdUpdateBuffer or vkCmdClearDepthStencilImage calls in this
backend; the depth/stencil clear helper draws through the existing Draw wrapper.
The extra pairs may change timing and can exhaust the same 8192-slot ring;
check dropped. Operations recorded outside a worker tick still do not allocate
GPU marker slots, since the worker timeline cannot retire their submissions.

All Queue wrapper Submit/Submit2, BindSparse and Present calls emit enter/return
events, correlated by the enter seq stored as call. Queue handles and caller
file/line distinguish worker, blit, DiscardFrame, present and TurboMode (which
owns a separate VkDevice/queue). Submit events include command-buffer handles,
wait/signal semaphores, and sync2 values/stages. DeviceWaitIdle also emits events.
Outside the worker, a submit/present event tick is a snapshot of latest recording
progress, not a claim that the operation belongs to that scheduler tick;
worker_tick=0 identifies external submits. Sparse sites supply CurrentTick and
report actual buffer/opaque-image/image bind totals, fence, waits and signals.
All four sparse sites also record entry/return around their existing fence wait.
No additional waits, mutexes around queue operations, or synchronization changes
are introduced. Events can be evicted from the existing finite event ring.

Read sparse-return result=0 as accepted, not completed; sparse-fence-return
result=0 is the completion evidence. A missing return is only meaningful when
the enter was retained and the trace has no relevant coverage loss. API returns
and CPU seq ordering never replace Vulkan execution dependencies.

## Case study: Radeon 890M, PREFLUSH and HAGS (2026-11-14)

Game: The Legend of Zelda: Ocarina of Time (0400486024DEC000), Kokiri Forest.
Environment reported by Oscar: Radeon 890M, Windows 11 build 26300, AMD proprietary
Vulkan drivers 2.0.395 and 2.0.406, High GPU accuracy (split inactive), asynchronous
CPU ASTC. The inspected HAGS-off logs identify driver 2.0.406. PREFLUSH and the
launcher environment remained enabled. The owner disabled Windows Hardware
Accelerated GPU Scheduling (reported HwSchMode=1) and rebooted; the emulator did
not change any system setting. HAGS state/reboot are owner observations, not
measurements made by this trace.

| Configuration | Observed outcome |
| --- | --- |
| HAGS on, PREFLUSH on | Owner reports 7/7 hangs near entry into Kokiri (~98–103 s). |
| HAGS off after reboot, PREFLUSH on | 2/2 runs without a hang, about 2.5 minutes in Kokiri each. |
| HAGS on, legacy submit + timeline experiment | Hung, GPU tick 14476 / wait 14477. |
| HAGS on, per-submit VkFence experiment | Hung, GPU tick 14680 / wait 14681. |

Both successful runs used the normal timeline + submit2 path with neither
experiment enabled. Their logs extend to approximately 275 and 269 seconds
since process logging began; that is distinct from time spent in Kokiri.
The temporary completion/submit experiments did not avoid the hang and have
been removed. There is no vendor-specific completion override in this change.

In the fully instrumented failing timeline run, tick 14275 had matching before
and after tokens for upload-end (seq 1823552) and main-end (1823553), with no
marker drops or open pairs. Submit2 accepted timeline value 14275 on semaphore
0x1dd9f634240 with ALL_COMMANDS; host completion stayed at 14274. No sparse,
present or device-idle event was found between the retained submits 14271–14277.
The legacy and VkFence runs likewise had no open marker pairs and dropped=0,
but their completion primitive did not advance. This is not a failure isolated
to the timeline API or the submit2 entry point.

Conclusion: strong evidence for an AMD driver/HAGS interaction under this
workload; disabling HAGS avoided the hang in the two observed runs. No emulator
defect causing this particular completion stall was established. This finite
A/B does not prove a universal driver root cause, permanent stability, or absence
of an emulator race masked by changed scheduling. PREFLUSH changes readback work
and CPU/GPU overlap; an increase in submit frequency was not measured.

Separate unresolved issue: the CPU-write versus downloaded-writeback window
identified during code review remains a candidate data race, not a proven cause
of this hang. The owner also reports corruption of the "BOSQUE KOKIRI" label with
PREFLUSH; stale data is a hypothesis, not an established explanation. No deferred
writes, coverage fix, host re-signaling, or marker-based resource retirement is
included in this diagnostic change.

Local evidence (kept on the owner's PC under Nova5/pruebas; not bundled):

- preflush-marcadores2-20261114-002115/novas_log.txt
- exp-legacy-submit-20261114-002906/novas_log.txt
- exp-master-fences-20261114-003141/novas_log.txt
- hags-off-preflush-20261114-005345/novas_log.txt
- hags-off-preflush-2-20261114-005852/novas_log.txt

References: [AMD marker command](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteBufferMarkerAMD.html),
[host timeline signal restrictions](https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphoreSignalInfo.html).
A marker observation must never authorize advancing the emulator's completion
tick, re-signaling a pending timeline value, recycling resources or publishing
readbacks. Only the normal Vulkan completion path does that.
