"""All-GT interpretation tables and scientific figures, saved-output CPU only."""
from c1_common import *
os.environ['MPLCONFIGDIR']=str(R/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
CONDITIONS=['G0S0','G0S1','G1S0','G1S1']

def run():
 assert json.loads((R/'validation.json').read_text())['passed']
 metrics=json.loads((R/'counterfactual_metrics.json').read_text());gt=list(csv.DictReader((R/'per_GT_counterfactual.csv').open()));idx={(int(x['sample_idx']),int(x['gt_idx']),x['condition']):x for x in gt}
 cases={}
 for sel in chosen():
  sid=int(sel['sample_idx'])
  for g in [0,1]:
   for s in [0,1]:cases[(sid,f'G{g}S{s}')]=load_case('baseline' if g==s else 'counterfactual',sid,g,s)[0]
 def csvout(name,rows):
  with (R/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 studies=[];keys=[];changed=[]
 for row in gt:
  if row['condition'] not in ['G0S1','G1S0'] or row['transition_from_original_same_geometry'] in ['TP_to_TP','FN_to_FN']:continue
  changed.append({k:row[k] for k in ['sample_idx','gt_idx','condition','group','transition_from_original_same_geometry','availability_class','matched_anchor_id','matched_anchor_score','count_ge70_raw_decoded','count_ge70_score_pass','count_ge70_nms_keep','surviving_strict_original_deletion_reasons']})
 for sel in chosen():
  if not sel['group'].startswith('loss_'):continue
  sid=int(sel['sample_idx']);gi=int(sel['gt_idx']);a0=cases[(sid,'G0S0')];b=cases[(sid,'G1S1')];n=cases[(sid,'G1S0')];mat0=a0['IoU_GT_raw'];mat1=b['IoU_GT_raw'];strict=np.flatnonzero(mat1[gi]>=.7);g0anchor=int(a0['GT_matched_anchor'][gi]);match=int(n['GT_matched_anchor'][gi])
  r={'sample_idx':sid,'gt_idx':gi,'G1S1_TP70':int(b['GT_TP70'][gi]),'G1S0_TP70':int(n['GT_TP70'][gi]),'restored':int(n['GT_TP70'][gi]),'G1S1_stage':idx[(sid,gi,'G1S1')]['availability_class'],'G1S0_stage':idx[(sid,gi,'G1S0')]['availability_class'],'G1_raw_strict_ids':json.dumps(strict.tolist()),'strict_pass_score_S1':json.dumps(strict[b['mask_score_pass'][strict]].tolist()),'strict_pass_score_S0':json.dumps(strict[n['mask_score_pass'][strict]].tolist()),'strict_final_S1':json.dumps(strict[b['mask_final_range'][strict]].tolist()),'strict_final_S0':json.dumps(strict[n['mask_final_range'][strict]].tolist()),'max_S0_on_G1_strict':float(n['scores_all'][strict].max()),'max_S1_on_G1_strict':float(b['scores_all'][strict].max()),'G0S0_matched_anchor':g0anchor,'IoU_G0_at_G0_match':float(mat0[gi,g0anchor]),'IoU_G1_at_G0_match':float(mat1[gi,g0anchor]),'restored_matched_anchor':match,'restored_anchor_S1':float(b['scores_all'][match]) if match>=0 else '', 'restored_anchor_S0':float(n['scores_all'][match]) if match>=0 else '', 'restored_anchor_IoU_G1':float(mat1[gi,match]) if match>=0 else '', 'restored_anchor_old_reason':REASONS[int(b['first_deletion_code'][match])] if match>=0 else ''}
  studies.append(r)
  relevant=set(strict.tolist())|{g0anchor}
  for a in [b,n]:relevant.update(int(x) for x in a['suppressor'][strict] if x>=0)
  for ai in sorted(relevant):
   rr={'sample_idx':sid,'gt_idx':gi,'anchor_id':ai,'score_S0':float(a0['scores_all'][ai]),'score_S1':float(b['scores_all'][ai]),'IoU_G0':float(mat0[gi,ai]),'IoU_G1':float(mat1[gi,ai])}
   for c in CONDITIONS:
    a=cases[(sid,c)];rr[c+'_first_deletion']=REASONS[int(a['first_deletion_code'][ai])];rr[c+'_suppressor']=int(a['suppressor'][ai]);rr[c+'_NMS_overlap']=float(a['suppressor_overlap'][ai]);rr[c+'_final_TP70']=int(a['anchor_final_TP70'][ai])
   keys.append(rr)
 csvout('key_case_summary.csv',studies);csvout('key_case_anchor_evidence.csv',keys);csvout('GT_changed_targets.csv',changed)
 # Every GT's all high-IoU candidates remain in per_GT_anchor_lineage.csv.
 figures=[]
 def finish(fig,name):
  for ext in ['png','pdf']:
   path=R/'figures'/(name+'.'+ext)
   if path.exists():raise FileExistsError(path)
   fig.savefig(path,dpi=180,bbox_inches='tight');figures.append(str(path.relative_to(R)))
  plt.close(fig)
 colors={'GT':'black','raw':'#aaaaaa','pass':'#348abd','match':'#168a45','other':'#d45b42'}
 def polygon(ax,box,color,ls='-',lw=1,alpha=1):
  xy=box[:4,:2];xy=np.vstack([xy,xy[0]]);ax.plot(xy[:,0],xy[:,1],color=color,ls=ls,lw=lw,alpha=alpha)
 for study in studies:
  sid=study['sample_idx'];gi=study['gt_idx'];truth=cases[(sid,'G0S0')]['GT_canonical'][gi];center=truth[:4,:2].mean(0)
  fig,axes=plt.subplots(2,2,figsize=(13,10),sharex=True,sharey=True)
  for ax,c in zip(axes.flat,CONDITIONS):
   a=cases[(sid,c)];mat=a['IoU_GT_raw'][gi];strict=np.flatnonzero(mat>=.7);scorepass=strict[a['mask_score_pass'][strict]];matched=int(a['GT_matched_anchor'][gi])
   for ai in strict:polygon(ax,a['raw_corners'][ai],colors['raw'],'--',.75,.6)
   for ai in scorepass:polygon(ax,a['raw_corners'][ai],colors['pass'],'-',1,.8)
   suppressors=set(int(x) for x in a['suppressor'][strict] if x>=0)
   near=set(int(x) for x in a['ids_final_range'] if mat[x]>0)|suppressors
   for ai in sorted(near):polygon(ax,a['raw_corners'][ai],colors['match'] if ai==matched else colors['other'],'-',2,.95)
   polygon(ax,truth,colors['GT'],'-',2.2,1)
   label=set(suppressors)
   if len(strict):label.add(int(strict[a['scores_all'][strict].argmax()]))
   if matched>=0:label.add(matched)
   for j,ai in enumerate(sorted(label)):
    xy=a['raw_corners'][ai,:4,:2].mean(0);color=colors['match'] if ai==matched else colors['other'] if ai in near else colors['pass'] if ai in scorepass else '#777777'
    ax.annotate(f"#{ai} s={a['scores_all'][ai]:.3f}\nIoU={mat[ai]:.3f}",xy=xy,xytext=(8,18+28*(j%3)),textcoords='offset points',fontsize=7,color=color,bbox={'facecolor':'white','edgecolor':'none','alpha':.8},arrowprops={'arrowstyle':'-','lw':.6,'color':color})
   row=idx[(sid,gi,c)];tp=bool(a['GT_TP70'][gi]);counts='/'.join(row['count_ge70_'+k] for k in STAGES)
   ax.set_title(f"{c}: {'TP70' if tp else 'FN70'} | {row['availability_class']}\nstrict R/S/Q/T/N/F: {counts}",fontsize=9)
   ax.set_xlim(center[0]-5.5,center[0]+5.5);ax.set_ylim(center[1]-5.5,center[1]+5.5);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('Ego x (m)');ax.set_ylabel('Ego y (m)')
  fig.suptitle(f'Fixed-anchor score crossover: sample {sid}, GT {gi}',fontsize=14)
  handles=[Line2D([0],[0],color=colors['GT'],lw=2,label='GT'),Line2D([0],[0],color=colors['raw'],ls='--',label='All raw IoU >= 0.7'),Line2D([0],[0],color=colors['pass'],label='Score-passing strict'),Line2D([0],[0],color=colors['match'],lw=2,label='Final matched anchor'),Line2D([0],[0],color=colors['other'],lw=2,label='Other final / NMS suppressor')]
  fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9);fig.tight_layout(rect=[0,.08,1,.95]);finish(fig,f'case_sample_{sid}_GT_{gi}')
 # Focal table contains all 18, including failures, gains and stable controls.
 sel=chosen();fig,ax=plt.subplots(figsize=(10,9));flags=np.array([[int(idx[(int(x['sample_idx']),int(x['gt_idx']),c)]['final_TP70']) for c in CONDITIONS] for x in sel])
 ax.imshow(flags,cmap=ListedColormap(['#f5b9b0','#b9dec6']),vmin=0,vmax=1,aspect='auto')
 short={'score_filter':'Score','nms_overlap_suppression':'NMS','no_high_iou_raw_candidate':'Raw','size_or_z_filter':'Quality','nms_top1000_truncation':'Top1000','post_nms_range_filter':'Range','high_iou_survives_final':'Match'}
 for i,x in enumerate(sel):
  for j,c in enumerate(CONDITIONS):
   row=idx[(int(x['sample_idx']),int(x['gt_idx']),c)];text='TP #'+row['matched_anchor_id'] if flags[i,j] else 'FN: '+short[row['availability_class']];ax.text(j,i,text,ha='center',va='center',fontsize=8)
 ax.set_xticks(range(4),CONDITIONS);ax.set_yticks(range(18),[f"{x['sample_idx']}/{x['gt_idx']} ({x['group'].split('_')[0]})" for x in sel]);ax.set_title('All 18 prespecified focal GTs (computational anchor IDs)');ax.set_xlabel('Fixed geometry / classification-score source');fig.tight_layout();finish(fig,'focal_GT_four_conditions')
 fig,axes=plt.subplots(1,2,figsize=(12,4.5));palette=['#246ba0','#46a075','#e68a42','#b04c7d']
 groups=['loss_TP70_to_FN70','gain_FN70_to_TP70','control_TP70_to_TP70']
 for j,c in enumerate(CONDITIONS):
  axes[0].bar(np.arange(3)+(j-1.5)*.18,[metrics['GT_group_metrics'][g][c]['TP70'] for g in groups],width=.18,color=palette[j],label=c)
  axes[1].bar(j,metrics['GT_group_metrics']['all_GT'][c]['TP70'],color=palette[j]);axes[1].text(j,metrics['GT_group_metrics']['all_GT'][c]['TP70']+1,str(metrics['GT_group_metrics']['all_GT'][c]['TP70']),ha='center')
 axes[0].set_xticks(range(3),['6 loss GTs','6 gain GTs','6 controls']);axes[0].set_ylim(0,6.7);axes[0].set_ylabel('TP70 count');axes[0].legend(ncol=2);axes[1].set_xticks(range(4),CONDITIONS);axes[1].set_ylim(0,162);axes[1].set_ylabel('TP70 count / 162');fig.suptitle('Outcome-selected 18 samples; counts are descriptive');fig.tight_layout();finish(fig,'TP70_group_counts')
 aggregate=json.loads((R/'aggregate_native_stats.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,5));rankrows=[]
 for ax,t in zip(axes,['50','70']):
  for j,c in enumerate(CONDITIONS):
   stat=aggregate[c][t];scores=np.array(stat['score']);order=np.argsort(-scores);tp=np.array(stat['tp'])[order];fp=np.array(stat['fp'])[order];rec=np.cumsum(tp)/stat['gt'];prec=np.cumsum(tp)/(np.cumsum(tp)+np.cumsum(fp));ap=metrics['metrics'][c][t]['AP_descriptive_selected18']
   ax.step(rec,prec,where='post',label=f'{c} AP={ap:.5f}',color=palette[j])
   for rank,i in enumerate(order):rankrows.append({'condition':c,'IoU_threshold':t,'rank':rank+1,'score':float(scores[i]),'TP':int(tp[rank]),'FP':int(fp[rank]),'recall':float(rec[rank]),'precision':float(prec[rank])})
  ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.set_xlim(0,1);ax.set_ylim(0,1.01);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(f'IoU {float(t)/100:.1f}')
 fig.suptitle('PR on outcome-selected 18 samples (not a 910-sample estimate)');fig.tight_layout();finish(fig,'PR_descriptive_selected18');csvout('PR_global_ranking.csv',rankrows)
 interpretation={'loss_case_studies':studies,'restored_loss_GT_count':sum(x['restored'] for x in studies),'unrestored_loss_GT_count':sum(not x['restored'] for x in studies),'all_fixed_geometry_GT_changes':changed,'figures':figures,'H1_supported_for_selected_cases':sum(x['restored'] for x in studies)>0,'H2_not_all_losses_restored':any(not x['restored'] for x in studies),'claims_limited_to_saved_head_counterfactual_and_selected18':True}
 save(R/'interpretation.json',interpretation)
 log('EVIDENCE_FIGURES_COMPLETE',figure_files=len(figures))
if __name__=='__main__':run()
