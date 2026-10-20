"""Retain the rejected first repair and its default-temperature failure."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
import torch
import run_model_proofs as proof
import fixed_model_v1_saturated as v1

data,args,patients=proof.load_inputs()
torch.manual_seed(17)
original=v1.build_model(args,'original').to(proof.DEVICE)
fixed=v1.build_model(args,'readout_fix').to(proof.DEVICE)
fixed.load_state_dict(original.state_dict(),strict=True)
row=proof.grad_row(fixed,data,1017)
result={'status':'rejected_first_repair_not_used_for_patient_training',
        'seed':17,'patients':patients,'v1_gradient':row,
        'criterion':'connected gradients must also be numerically nonzero',
        'original_proof_failure':'AssertionError: nonzero_tensor_count >= 4',
        'fixed_model_v1_sha256':hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest()}
path=Path(__file__).resolve().parent/'model/v1_rejected.json'
path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
