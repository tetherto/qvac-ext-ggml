# Allocation-free buffer sizing

`ggml_backend_buft_get_buffer_alloc_size(buft, payload_bytes)` predicts the
size reported by a single buffer allocated for that payload request. The
optional registry procedure of the same name implements the conversion;
without it the conversion is the identity. It does not change the backend
interface layout or the tensor-level `get_alloc_size` contract.

The callback must be allocation-free, must not mutate backend state, and must
return at least the requested payload size. `SIZE_MAX` denotes an
unrepresentable request. A zero request always reports zero: the public
allocation wrapper returns an empty metadata buffer without invoking the
backend allocator. This value describes backend-reported buffer bytes, not
unreported driver heaps, process RSS or all physical memory in use.

Context sizing and graph reservation apply this conversion to each of the
actual allocator's payload chunks, then add the results with saturation. The
number of chunks comes from normal packing and liveness analysis; it is not
inferred from total bytes or charged once per component. Repeated graph
buffer types share an allocator and are charged only at their first index,
matching the allocated-buffer query. Tensor payloads are never allocated by
the sizing functions.

The Hexagon implementation shares one checked guard-size helper between
prediction and both default/repack buffer allocators. Each nonempty buffer
reports its payload plus the guard. Graph growth compares the predicted new
buffer size with the existing reported buffer size so that a small payload
increase cannot silently consume the retained guard. The shared-buffer
detachment check uses the same conversion, keeping another scheduler's
retained storage alive when growth fits within the old guard allowance.

`test-backend-buffer-size` uses a host-portable synthetic backend with a
small payload cap and a non-page guard. It checks fragmented context/graph
allocations against actual buffers, repeated buffer types, views, empty and
already-allocated contexts, zero-size dummy buffers, guard overflow,
saturating totals and graph growth within the old buffer's guard allowance.
Shared scheduler growth also checks source-payload preservation and exact
buffer lifetimes after detachment.
The existing CPU memory regression continues to check identity-backend byte
parity.

Hexagon's `memory_unified` property states that its FastRPC tensor buffers
compete with host allocations for system RAM. Its zero device-capacity query
still means that no separate device-specific capacity estimate is available;
clients can use their system-RAM policy for that shared pool. This does not
claim that every available byte can be allocated through FastRPC or mapped
at once, nor replace per-buffer/mapping limits or a driver memory margin.
