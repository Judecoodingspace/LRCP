"""P4 CPU attribution contract DRAFT; no inference entry point, torch, or LRCP imports.
Candidate indices encode anchor locations, never object identities.
Requires externally captured native stage lineage and exact float32 BEV IoU.
"""
import numpy as np

STAGES = ('raw_decoded', 'score_pass', 'quality_pass', 'nms_top1000', 'nms_keep', 'final_range')
CAUSES = ('no_high_iou_raw_candidate', 'score_filter', 'size_or_z_filter',
          'nms_top1000_truncation', 'nms_overlap_suppression', 'post_nms_range_filter')


def attribute_gt(iou_gt_candidate, stage_candidate_ids, threshold=0.7):
    """Return per-GT stage availability; not TP/FN and not causal fractions.

    iou_gt_candidate: G x 12800 (or actual captured K), exact same geometry
      at all stages, float32. Direction correction/projection already applied
      to an independent all-anchor decode; native retained rows must agree.
    stage_candidate_ids: ordered native raw anchor IDs, preserving NMS order.
    Nonfinite IoU aborts; missing stages abort rather than inventing a cause.
    """
    iou = np.asarray(iou_gt_candidate)
    if iou.ndim != 2 or not np.isfinite(iou).all():
        raise ValueError('Invalid IoU; cannot classify generation as missing.')
    g, k = iou.shape
    stages = []
    prior = None
    for name in STAGES:
        ids = np.asarray(stage_candidate_ids[name], dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= k):
            raise ValueError('Invalid or nonunique lineage: '+name)
        if prior is not None and not set(ids).issubset(set(prior)):
            raise ValueError('Non-nested stage lineage: '+name)
        stages.append(ids)
        prior = ids
    if set(stages[0]) != set(range(k)):
        raise ValueError('Raw stage must include EVERY decoded anchor, including low scores.')
    result=[]
    for gi in range(g):
        counts=[int(np.count_nonzero(iou[gi,ids] >= threshold)) for ids in stages]
        maxima=[float(iou[gi,ids].max()) if len(ids) else 0.0 for ids in stages]
        first=next((n for n,c in enumerate(counts) if c==0),None)
        # Record every high-IoU candidate's first deletion, not just one argmax.
        deleted={}
        for si in range(1,len(stages)):
            dropped=set(stages[si-1])-set(stages[si])
            deleted[CAUSES[si]]=sorted(int(cid) for cid in dropped if iou[gi,cid] >= threshold)
        result.append({'gt_idx':gi,'high_iou_count':dict(zip(STAGES,counts)),
                       'best_iou':dict(zip(STAGES,maxima)),
                       'availability_class':'high_iou_survives_final' if first is None else CAUSES[first],
                       'high_iou_deleted_ids':deleted})
    return result


def replay_native_nms(corners, scores, threshold, convert_format, compute_iou):
    """CPU shadow of box_utils.nms_rotated, with native float dtype/sort/top1000.
    Must compare keep IDs to native call. This function never replaces it.
    Returns suppressor lineage and separate top1000 truncation.
    """
    corners=np.asarray(corners);scores=np.asarray(scores)
    if len(scores) != len(corners):raise ValueError('Length mismatch')
    polygons=convert_format(corners)
    order=scores.argsort()[::-1]
    excluded=order[1000:].tolist()
    pending=order[:1000].copy();picks=[];suppressed=[]
    while len(pending):
        chosen=int(pending[0]);picks.append(chosen)
        iou=compute_iou(polygons[chosen],polygons[pending[1:]])
        positions=np.where(iou>threshold)[0]+1
        suppressed.extend({'removed_nms_input_idx':int(pending[p]),
                           'suppressor_nms_input_idx':chosen,
                           'bev_iou_float32':float(iou[p-1])} for p in positions)
        pending=np.delete(pending,positions)
        pending=np.delete(pending,0)
    return {'keep':np.asarray(picks,dtype=np.int32),'top1000_excluded':excluded,'suppressed':suppressed,
            'tied_score_entries':int(len(scores)-len(np.unique(scores)))}


def paired_capture_hooks(model, state, emit):
    """DRAFT opt-in factory; caller owns model/eval/no_grad and approved pilot.
    No hook installation until explicitly called. No device allocation here.
    emit(name,tensor,metadata) MUST detach/copy synchronously to bounded CPU
    storage, not retain mutable device aliases or modify tensors.
    No complete collector/PPU launcher is supplied in this precheck.
    """
    handles=[]
    def scale(module, inputs, output):
        if output.ndim != 4 or output.shape[-1] != 2:raise ValueError('Unexpected MetaFlow shape')
        effective=output*state['alpha']
        emit('metaflow_raw',output,{'slot':state['remote_slot'],'alpha':state['alpha']})
        emit('metaflow_effective',effective,{'slot':state['remote_slot'],'alpha':state['alpha']})
        return effective
    # Caller must set remote_slot from enclosing forward's calibrate_data
    # iteration (invalid slots do not invoke MetaFlow); do NOT count calls as CAV IDs.
    handles.append(model.meta_flow.register_forward_hook(scale))
    def fusion_pre(module,args):
        emit('fusion_input_offsets',args[4],{'alpha':state['alpha']})
        for level,feat in enumerate(args[0]):
            emit('fusion_input_features_summary',feat,{'level':level,'alpha':state['alpha']})
    handles.append(model.backbone.defor_encoder.register_forward_pre_hook(fusion_pre))
    def fusion_post(module,args,out):
        emit('fused_features_summary_and_roi',out,{'alpha':state['alpha']})
    handles.append(model.backbone.defor_encoder.register_forward_hook(fusion_post))
    for name in ('cls_head','reg_head','dir_head'):
        def capture(module,args,out,name=name):emit(name,out,{'alpha':state['alpha']})
        handles.append(getattr(model,name).register_forward_hook(capture))
    return handles  # caller removes ALL in finally; no persistent patch
