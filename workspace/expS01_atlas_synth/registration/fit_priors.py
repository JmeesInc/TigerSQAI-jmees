"""Export shrinkage prior proposals from accepted Stage 1 retrievals. Not ground-truth calibration."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
import yaml
from render.util import atomic_json
from render.camera import basis,unit


def regularized_normal(spec,values,strength,min_sd_fraction):
    values=np.asarray(values,float);n=len(values);blend=n/(n+strength)
    out=deepcopy(spec);mean=(1-blend)*spec['mean']+blend*float(values.mean())
    variance=(1-blend)*(spec['sd']**2+(spec['mean']-mean)**2)+blend*float(np.mean((values-mean)**2))
    out['mean']=float(np.clip(mean,spec['min'],spec['max']));out['sd']=max(float(np.sqrt(variance)),spec['sd']*min_sd_fraction)
    return out


def categorical(spec,values,strength):
    out=deepcopy(spec);prior=np.array(spec['weights'],float);prior/=prior.sum();counts=np.array([values.count(v) for v in spec['values']]);out['weights']=((counts+strength*prior)/(counts.sum()+strength)).tolist();return out


def fit(results,camera,dissection,cfg):
    accepted=[r for r in results if r.get('accepted')];settings=cfg['prior_fit']
    histogram=Counter(r['top_k'][0]['bank_id'] for r in accepted)
    if len(accepted)<settings['minimum_accepted_frames']:raise ValueError('Too few accepted frames; priors unchanged')
    if len(histogram)<settings['minimum_unique_bank_poses']:raise ValueError('Too few unique retrieved poses; expand bank')
    if max(histogram.values())/len(accepted)>settings['maximum_single_pose_fraction']:raise ValueError('Retrieval collapsed onto one bank pose; expand bank')
    # Retain per-real-frame weights (selection distribution), report duplicate/ESS limits.
    c=deepcopy(camera);d=deepcopy(dissection);strength=settings['shrinkage_pseudocount'];floor=settings['minimum_sd_fraction_of_original']
    poses=[r['camera'] for r in accepted]
    for configkey,posekey in [('roll_deg','sensor_roll_deg'),('horizontal_fov_deg','horizontal_fov_deg')]:
        c['scope'][configkey]=regularized_normal(c['scope'][configkey],[p[posekey] for p in poses],strength,floor)
    log=np.log([p['working_distance_mm'] for p in poses]);spec=c['scope']['working_distance_mm'];normal={'mean':spec['log_mean'],'sd':spec['log_sd'],'min':np.log(spec['min']),'max':np.log(spec['max'])};new=regularized_normal(normal,log,strength,floor);spec.update(log_mean=new['mean'],log_sd=new['sd'])
    c['scope']['oblique_angle_deg']=categorical(c['scope']['oblique_angle_deg'],[p['oblique_angle_deg'] for p in poses],strength)
    names=c['targets']['names'];target=categorical({'values':names,'weights':c['targets']['weights']},[p['target_name'] for p in poses],strength);c['targets']['weights']=target['weights']
    coupled=[p for p in poses if p.get('window_coupling')]
    if coupled and c.get('window_coupling',{}).get('enabled',False):
        wc=c['window_coupling']
        wc['short_side_occupancy']=regularized_normal(wc['short_side_occupancy'],[p['window_coupling']['requested_short_side_occupancy'] for p in coupled],strength,floor)
        wc['center_target_probability']=float((sum(p['target_name'].startswith('window:') for p in coupled)+strength*wc['center_target_probability'])/(len(coupled)+strength))
        anchors=list(wc['anchor_weights'])
        weights=categorical({'values':anchors,'weights':[wc['anchor_weights'][a] for a in anchors]},[p['window_coupling']['window']['anchor'] for p in coupled],strength)
        wc['anchor_weights']=dict(zip(anchors,weights['weights']))
    # Derive the inverse-sampler meridian angle, distinct from actual shaft axial rotation.
    azimuth=[]
    for p in poses:
        direction=unit(np.array(p['target_mm'])-p['port_mm']);a,b=basis(direction);shaft=np.array(p['shaft_axis_ras']);meridian=-(shaft-direction*(shaft@direction))
        if np.linalg.norm(meridian)>1e-7:azimuth.append(float(np.arctan2(meridian@b,meridian@a)))
    if azimuth:
        old=c['scope']['shaft_azimuth_rad'];from scipy.special import i0e,i1e
        resultant=np.sum(np.exp(1j*np.array(azimuth)))+strength*(i1e(old['kappa'])/i0e(old['kappa']))*np.exp(1j*old['mean']);r=abs(resultant)/(len(azimuth)+strength)
        k=2*r+r**3+5*r**5/6 if r<.53 else (-.4+1.39*r+.43/(1-r) if r<.85 else 1/(r**3-4*r*r+3*r))
        old.update(mean=float(np.angle(resultant)),kappa=float(min(k,20.)))
    for key in ['k1','k2']:
        c['distortion'][key]=regularized_normal(c['distortion'][key],[p['distortion'].get(key,0.) for p in poses],strength,floor)
    for key,extract in [('fy_over_fx',lambda p,r:p['K'][1][1]/p['K'][0][0]),('principal_x_offset_fraction',lambda p,r:(p['K'][0][2]-(r['resolution'][0]-1)/2)/r['resolution'][0]),('principal_y_offset_fraction',lambda p,r:(p['K'][1][2]-(r['resolution'][1]-1)/2)/r['resolution'][1])]:
        c['intrinsics'][key]=regularized_normal(c['intrinsics'][key],[extract(p,r) for p,r in zip(poses,accepted)],strength,floor)
    if c['intrinsics']['focal_x_px'] is not None:
        c['intrinsics']['focal_x_px']=regularized_normal(c['intrinsics']['focal_x_px'],[p['K'][0][0]*c['resolution'][0]/r['resolution'][0] for p,r in zip(poses,accepted)],strength,floor)
    t=np.clip([r['progress'] for r in accepted],1e-6,1-1e-6);old=d['progress'];lo=old.get('min',0.);hi=old.get('max',1.);initial=np.log([old['alpha'],old['beta']])
    def loss(x):
        a,b=np.exp(x);norm=beta.cdf(hi,a,b)-beta.cdf(lo,a,b)
        return float(-np.sum(beta.logpdf(t,a,b)-np.log(max(norm,1e-300)))+strength*np.sum((x-initial)**2))
    optimized=minimize(loss,initial,bounds=[(-2.3,4.6)]*2,method='L-BFGS-B')
    if not optimized.success:raise ValueError('Truncated beta fit failed; priors unchanged')
    old['alpha'],old['beta']=map(float,np.exp(optimized.x))
    # Presence identifies zero tools, NOT the count of shafts/components when tools exist.
    presence=[r['observed_presence_ids'] for r in accepted];oldcount=d['instruments']['count'];values=oldcount['values'];w=np.array(oldcount['weights']);w=w/w.sum();zero=values.index(0);p0=(sum(1 not in x for x in presence)+strength*w[zero])/(len(presence)+strength);others=np.arange(len(w))!=zero;w[others]*=(1-p0)/w[others].sum();w[zero]=p0;oldcount['weights']=w.tolist()
    for name,cid in [('blood',24),('resection',25)]:d[name]['probability']=float((sum(cid in x for x in presence)+strength*d[name]['probability'])/(len(presence)+strength))
    report={'accepted_frames':len(accepted),'unique_bank_poses':len(histogram),'effective_pose_count':float(len(accepted)**2/sum(v*v for v in histogram.values())),'retrieval_counts':dict(histogram),'status':'prior_proposal_requires_rendered_batch_comparison','unchanged':'port geometry/ranges, patient anatomy, physical bounds; no patient coordinates inferred','nonidentifiability':'single label maps do not uniquely determine pose, FOV, anatomy or dissection; retrieved proposal is bank-conditioned','state_priors':'blood/resection probabilities are visibility-derived proposal values; instrument positive-count split retained, not inferred from connected components'}
    return c,d,report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--camera-prior',type=Path,default=Path('configs/camera_prior.yaml'));p.add_argument('--dissection',type=Path,default=Path('configs/dissection.yaml'));p.add_argument('--config',type=Path,default=Path('configs/registration.yaml'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Fit output must be empty')
    results=[json.loads(p.read_text()) for p in sorted((a.registration/'poses').glob('*.json'))]
    c,d,r=fit(results,yaml.safe_load(a.camera_prior.read_text()),yaml.safe_load(a.dissection.read_text()),yaml.safe_load(a.config.read_text()));a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'camera_prior.yaml').write_text(yaml.safe_dump(c,sort_keys=False));(a.output/'dissection.yaml').write_text(yaml.safe_dump(d,sort_keys=False));atomic_json(a.output/'fit_report.json',r);print(a.output)


if __name__=='__main__':main()
