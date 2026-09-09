"""Do not build a search bank until Round 4 aggregate and class-coverage gates pass."""
import json
from pathlib import Path
from copy import deepcopy


def require_report(path,configuration=None):
    if path is None:raise ValueError('Bank blocked: provide a >=128-frame calibration comparison.json with all four criteria and IDs 12/8/9 passing')
    report=json.loads(Path(path).read_text());gate=report.get('acceptance',{})
    if gate.get('schema_version')!=2 or not gate.get('bank_generation_allowed',False) or not gate.get('batch_complete',False) or gate.get('frames',0)<128 or gate.get('missing_required_bank_class_ids',[12,8,9]):
        raise ValueError('Bank blocked by Round 4 calibration/coverage gate')
    if configuration is not None:
        def canonical(value):
            value=deepcopy(value)
            for k in ['seed','device','threads_per_worker','patient_id','max_attempts_per_frame','render_timeout_seconds']:
                value['camera'].pop(k,None)
            # JSON normalizes YAML integer mapping keys before sorting.
            return json.dumps(json.loads(json.dumps(value)),sort_keys=True)
        reference=report.get('render_configuration')
        if reference is None or canonical(reference)!=canonical(configuration):
            raise ValueError('Bank configuration differs from passing calibration batch; rerender/re-evaluate these settings first')
    return gate
