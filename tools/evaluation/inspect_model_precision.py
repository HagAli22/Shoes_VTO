import onnx
from onnx import TensorProto
from pathlib import Path
from collections import Counter

models = [
    'deliverables/qualcomm_experiments/models/rtmpose_13kp_hand_improved5_opset12.onnx',
    'deliverables/qualcomm_experiments/models/rtmpose_13kp_hand_improved5_fp16.onnx',
    'deliverables/qualcomm_experiments/models/nanodet_stage_a_improved5_opset12.onnx',
    'deliverables/qualcomm_experiments/models/nanodet_stage_a_improved5_fp16.onnx',
]

def type_name(dt):
    try:
        return TensorProto.DataType.Name(dt)
    except Exception:
        return str(dt)

for m_path in models:
    p = Path(m_path)
    if not p.exists():
        print(f"File not found: {p}")
        continue
    print("=" * 80)
    print(f"MODEL: {p.name}")
    print(f"File Size: {p.stat().st_size:,} bytes ({p.stat().st_size / (1024*1024):.2f} MB)")
    
    model = onnx.load(str(p), load_external_data=False)
    graph = model.graph
    
    # 1. Opset
    opsets = {op.domain or 'ai.onnx': op.version for op in model.opset_import}
    print(f"Opset: {opsets}")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    
    # 2. Inputs & Outputs
    print("\nInputs:")
    for inp in graph.input:
        tt = inp.type.tensor_type
        shape = [d.dim_value if d.dim_value > 0 else (d.dim_param or '?') for d in tt.shape.dim]
        print(f"  - {inp.name}: {type_name(tt.elem_type)}, shape={shape}")
        
    print("Outputs:")
    for out in graph.output:
        tt = out.type.tensor_type
        shape = [d.dim_value if d.dim_value > 0 else (d.dim_param or '?') for d in tt.shape.dim]
        print(f"  - {out.name}: {type_name(tt.elem_type)}, shape={shape}")
        
    # 3. Initializers / Weights
    init_types = Counter()
    init_elements = Counter()
    for init in graph.initializer:
        dt = init.data_type
        tname = type_name(dt)
        init_types[tname] += 1
        num_elem = 1
        for d in init.dims:
            num_elem *= d
        init_elements[tname] += num_elem
        
    print("\nInitializers (Weights):")
    for dt_name, count in init_types.items():
        print(f"  - {dt_name}: {count} tensors, {init_elements[dt_name]:,} total elements")
        
    # 4. Node Types & Cast operations
    node_ops = Counter(n.op_type for n in graph.node)
    print(f"\nTotal Nodes: {len(graph.node)}")
    print(f"Top Operations: {node_ops.most_common(10)}")
    
    # Cast operations
    cast_nodes = [n for n in graph.node if n.op_type == 'Cast']
    print(f"Cast Nodes Count: {len(cast_nodes)}")
    if cast_nodes:
        cast_targets = Counter()
        for c in cast_nodes:
            for attr in c.attribute:
                if attr.name == 'to':
                    cast_targets[type_name(attr.i)] += 1
        print(f"  Cast Target Types: {dict(cast_targets)}")
        for idx, c in enumerate(cast_nodes[:5]):
            target_t = 'unknown'
            for attr in c.attribute:
                if attr.name == 'to':
                    target_t = type_name(attr.i)
            print(f"  Cast #{idx+1}: {c.input} -> {c.output} (to: {target_t})")
            
    # Quantization operations
    q_ops = [n.op_type for n in graph.node if any(kw in n.op_type.lower() for kw in ['quant', 'int8', 'qlinear', 'integer'])]
    print(f"Quantization Nodes: {Counter(q_ops) if q_ops else 'None'}")
    
    # Value Info types (intermediate tensors)
    vi_types = Counter()
    for vi in graph.value_info:
        if vi.type.HasField('tensor_type'):
            vi_types[type_name(vi.type.tensor_type.elem_type)] += 1
    if vi_types:
        print(f"Intermediate ValueInfo Data Types: {dict(vi_types)}")
    else:
        print("Intermediate ValueInfo Data Types: None stored in graph")

