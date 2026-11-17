import ast,hashlib,json,os
from pathlib import Path
R=Path(__file__).resolve().parent;S=Path('/workspace/LRCP');files=['opencood/models/meta_flow.py','opencood/models/point_pillar_deformable.py','opencood/models/sub_modules/defor_bev_backbone_resnet.py','opencood/models/sub_modules/defor_encoder_multi_scale.py','opencood/data_utils/post_processor/voxel_postprocessor.py','opencood/utils/box_utils.py','data_preprocess_tools/flow_label_gen_v2x-smi.py','opencood/logs/v2xsim_point_pillar_lidar_deformable_attn_2026_09_03_21_27_18/config.yaml']
hashes={}
for f in files:
 p=S/f;raw=p.read_bytes();hashes[str(p)]=hashlib.sha256(raw).hexdigest()
 if f.endswith('.py'):ast.parse(raw);compile(raw,str(p),'exec',dont_inherit=True)
p=S/files[3];tree=ast.parse(p.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DeforAttn');f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward');calls=[x for x in ast.walk(f) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='apply' and isinstance(x.func.value,ast.Name) and x.func.value.id=='MSDeformAttnFunction'];assert len(calls)==1
x=calls[0];assert [v.id for v in x.args[:5] if isinstance(v,ast.Name)]==['value','spatial_shapes','level_start_index','sampling_locations','attention_weights']
for k in ['attention_capture_draft.py','test_observer_draft.py','select_verify.py','cpu_contract_tests.py','verify_selected_heads.py','verify_semantic_hashes.py']:
 p=R/k;compile(p.read_bytes(),str(p),'exec',dont_inherit=True)
(R/'SOURCE_STATIC.json').write_text(json.dumps({'passed':True,'source_hashes':hashes,'kernel_apply_line':x.lineno,'kernel_argument_identity_verified':True,'observer_target_line':x.lineno,'source_compile_no_import':True,'no_real_model_forward':True,'note':'These are static interface checks, not an executable or READY PPU collector'},indent=2))
