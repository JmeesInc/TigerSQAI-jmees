"""Round 4 anatomy-only metrics. All six conditions are enforced conservatively."""
def evaluate(synthetic,reference,cfg):
    area_targets={r['fine_id']:r['mean_area_pct'] for r in reference['classes']}
    actual={'presence_absolute_error_sum_pp':sum(abs(synthetic['presence_pct'][r['fine_id']]-r['presence_pct']) for r in reference['classes']),
            'area_absolute_error_sum_pp':sum(abs(synthetic['mean_area_pct'][r['fine_id']]-r['mean_area_pct']) for r in reference['classes']),
            'pleura_mean_area_pct':float(synthetic['mean_area_pct'][10]),'lung_mean_area_pct':float(synthetic['mean_area_pct'][18]),
            'visible_class_count_median':synthetic['visible_class_count']['median'],'background_median_pct':synthetic['background_summary']['median_pct']}
    targets={'presence_absolute_error_sum_pp':0.,'area_absolute_error_sum_pp':0.,'pleura_mean_area_pct':area_targets[10],'lung_mean_area_pct':area_targets[18],'visible_class_count_median':reference['visible_class_count']['median'],'background_median_pct':reference['background_summary']['target_interval_pct']}
    def distance(value,target):
        return max(target[0]-value,0,value-target[1]) if isinstance(target,list) else abs(value-target)
    improved={k:distance(v,targets[k])<distance(cfg['baseline'][k],targets[k]) for k,v in actual.items()}
    # The final background requirement is an absolute interval, not merely improvement.
    improved['background_median_pct']=distance(actual['background_median_pct'],targets['background_median_pct'])==0
    enough=synthetic['frames']>=cfg['minimum_frames'];missing=[i for i in cfg['required_bank_class_ids'] if synthetic['presence_pct'][i]<=0]
    passed=bool(enough and all(improved.values()) and synthetic.get('empty_anatomy_frames',0)==0)
    return {'schema_version':2,'frames':synthetic['frames'],'minimum_frames':cfg['minimum_frames'],'baseline':cfg['baseline'],'targets':targets,'actual':actual,'improved_toward_target':improved,'acceptance_passed':passed,'status':'insufficient_sample' if not enough else ('passed' if passed else 'failed'),'required_bank_class_ids':cfg['required_bank_class_ids'],'missing_required_bank_class_ids':missing,'bank_generation_allowed':bool(passed and not missing)}
