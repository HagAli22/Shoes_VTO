import json

with open('deliverables/qualcomm_experiments/profiling/fp32_vs_fp16_quadro_p2000_profile.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for mname, mdata in d['models'].items():
    print('========================================================')
    print('MODEL:', mname)
    prof = mdata['profiling']
    print(f"Node count: CUDA = {prof['cuda_nodes']}, CPU = {prof['cpu_nodes']}")
    print('Provider Durations:', prof['provider_breakdown_ms'])
    print('Top Ops:')
    for op in prof['top_ops'][:5]:
        print(f"  {op['op']:18s} count={op['count']:4d} ms={op['total_ms']:8.3f} prov={op['provider']}")
    print('Slowest Nodes:')
    for n in prof['top_slowest_nodes'][:5]:
        print(f"  {n['op']:12s} {n['dur_ms']:7.3f} ms | {n['name']}")
    b1 = mdata['bench_b1_standard']
    b2 = mdata['bench_b2_standard']
    print(f"Batch 1 GPU Mean: {b1['gpu_mean_ms']:.2f} ms (P50: {b1['gpu_p50_ms']:.2f}, P95: {b1['gpu_p95_ms']:.2f})")
    print(f"Batch 2 GPU Mean: {b2['gpu_mean_ms']:.2f} ms (P50: {b2['gpu_p50_ms']:.2f}, P95: {b2['gpu_p95_ms']:.2f})")
    if 'gpu_mean_ms' in mdata.get('bench_b2_io_binding', {}):
        print(f"Batch 2 IO-Binding Mean: {mdata['bench_b2_io_binding']['gpu_mean_ms']:.2f} ms")
    if 'gpu_mean_ms' in mdata.get('bench_b2_cuda_graph', {}):
        print(f"Batch 2 CUDA Graph Mean: {mdata['bench_b2_cuda_graph']['gpu_mean_ms']:.2f} ms")
    tr = mdata['transfers_and_prep']
    print(f"Prep: {tr['preprocessing_mean_ms']:.3f} ms, H2D: {tr['h2d_copy_mean_ms']:.3f} ms, D2H: {tr['d2h_copy_mean_ms']:.3f} ms")
    mem = mdata['memory']
    print(f"Memory Peak: {mem['torch_peak_mem_mb']} MB | SMI: {mem['nvidia_smi_snapshot']}")

