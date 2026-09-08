"""Round 3: jointly evaluate the four requested aggregate measures."""

def evaluate(synthetic,reference,cfg):
    presence=sum(abs(float(synthetic['presence_pct'][r['fine_id']])-r['presence_pct']) for r in reference['classes'])
    actual={'presence_absolute_error_sum_pp':presence,'pleura_mean_area_pct':float(synthetic['mean_area_pct'][10]),'visible_class_count_median':synthetic['visible_class_count']['median'],'background_median_pct':synthetic['background_summary']['median_pct']}
    targets={'presence_absolute_error_sum_pp':0.,'pleura_mean_area_pct':next(r['mean_area_pct'] for r in reference['classes'] if r['fine_id']==10),'visible_class_count_median':reference['visible_class_count']['median'],'background_median_pct':reference['background_summary']['median_pct']}
    improved={k:abs(v-targets[k])<abs(cfg['baseline'][k]-targets[k]) for k,v in actual.items()}
    enough=synthetic['frames']>=cfg['minimum_frames']
    missing=[i for i in cfg['required_bank_class_ids'] if synthetic['presence_pct'][i]<=0]
    return {'frames':synthetic['frames'],'minimum_frames':cfg['minimum_frames'],'baseline':cfg['baseline'],'targets':targets,'actual':actual,'improved_toward_target':improved,'acceptance_passed':bool(enough and all(improved.values())),'status':'insufficient_sample' if not enough else ('passed' if all(improved.values()) else 'failed'),'required_bank_class_ids':cfg['required_bank_class_ids'],'missing_required_bank_class_ids':missing,'bank_generation_allowed':bool(enough and all(improved.values()) and not missing)}
