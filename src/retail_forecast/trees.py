from .common import *
from lightgbm import LGBMRegressor,early_stopping,log_evaluation
from xgboost import XGBRegressor
NAMES=['LightGBM','XGBoost','LightGBM + history','XGBoost + history','Seasonal naive']
def origin_features(history,future,scenario):
 assert history.ds.max()<future.ds.min()
 values=masked_y(history).ffill();assert values.tail(28).notna().all()
 X=make_features(future,scenario).copy()
 for label,value in {'origin_last':values.iloc[-1],'origin_lag7':values.iloc[-8],
                     'origin_mean7':values.tail(7).mean(),'origin_mean28':values.tail(28).mean(),
                     'origin_valid_fraction28':float(valid_labels(history.tail(28)).mean())}.items():X[label]=float(value)
 X['horizon_day']=np.arange(1,len(future)+1);return X

def origin_training(history,scenario):
 xs=[];ys=[]
 # One sample group per week; all labels must end before the fitting cutoff.
 for origin in range(ENCODER_LENGTH-1,len(history)-HORIZON,7):
  h=history.iloc[:origin+1];f=history.iloc[origin+1:origin+1+HORIZON];valid=valid_labels(f)
  X=origin_features(h,f,scenario);xs.append(X.loc[valid]);ys.append(f.loc[valid,'y'])
 assert xs
 return pd.concat(xs,ignore_index=True),pd.concat(ys,ignore_index=True)

def fit_tree(history,scenario,name,window):
 use_history=' + history' in name;kind=name.split(' +')[0];core=history.iloc[:-HORIZON];v=history.iloc[-HORIZON:];valid=valid_labels(v)
 if use_history:Xc,yc=origin_training(core,scenario);Xv=origin_features(core,v,scenario).loc[valid];Xf,yf=origin_training(history,scenario)
 else:
  mc=valid_labels(core);mf=valid_labels(history);Xc=make_features(core,scenario).loc[mc];yc=core.loc[mc,'y'];Xv=make_features(v,scenario).loc[valid];Xf=make_features(history,scenario).loc[mf];yf=history.loc[mf,'y']
 yv=v.loc[valid,'y']
 if kind=='LightGBM':
  pars=dict(learning_rate=.04,num_leaves=15,min_child_samples=10,reg_lambda=1.,random_state=SEED,n_jobs=CPU_THREADS,verbosity=-1)
  candidate=LGBMRegressor(n_estimators=600,**pars).fit(Xc,yc,eval_set=[(Xv,yv)],eval_metric='mae',callbacks=[early_stopping(40,verbose=False),log_evaluation(0)])
  rounds=int(candidate.best_iteration_ or 600);final=LGBMRegressor(n_estimators=rounds,**pars)
 else:
  pars=dict(objective='reg:squarederror',learning_rate=.04,max_depth=3,min_child_weight=5,reg_lambda=1.,subsample=1.,colsample_bytree=1.,tree_method='hist',random_state=SEED,n_jobs=CPU_THREADS,eval_metric='mae')
  candidate=XGBRegressor(n_estimators=600,early_stopping_rounds=40,**pars).fit(Xc,yc,eval_set=[(Xv,yv)],verbose=False)
  rounds=int(candidate.best_iteration+1);final=XGBRegressor(n_estimators=rounds,**pars)
 final.fit(Xf,yf)
 return {'model':final,'use_history':use_history},{'selected_rounds':rounds,'validation_MAE':float(abs(candidate.predict(Xv)-yv).mean()),'training_rows':len(yf),'history_features':use_history}
def predict_tree(bundle,history,future,scenario,name):
 features=origin_features(history,future,scenario) if bundle['use_history'] else make_features(future,scenario)
 # Extreme future-target replacement must not change a single input feature.
 changed=future.copy();changed['y']=12345.
 checked=origin_features(history,changed,scenario) if bundle['use_history'] else make_features(changed,scenario)
 pd.testing.assert_frame_equal(features,checked)
 return bundle['model'].predict(features)
def explain_trees(finals):
 out=RUNS_DIR/'trees';rows=[];sparse=[]
 for scenario,frame in datasets.items():
  for name in NAMES[:-1]:
   m=finals[(scenario,name)]['model'];columns=list(m.feature_names_in_)
   gains=m.booster_.feature_importance(importance_type='gain') if name.startswith('LightGBM') else m.feature_importances_
   for col,value in zip(columns,gains):rows.append({'scenario':scenario,'model':name,'feature':col,'importance':float(value),'definition':'split_gain' if name.startswith('LightGBM') else 'normalized_gain'})
   if name.startswith('LightGBM'):m.booster_.save_model(str(out/f'{scenario}_{name.replace(" + history","_history")}.txt'))
   else:m.save_model(out/f'{scenario}_{name.replace(" + history","_history")}.json')
  X=make_features(frame.iloc[:TRAIN_END+1],scenario);valid=valid_labels(frame.iloc[:TRAIN_END+1])
  for col in X.columns:
   if col.startswith('holiday_'):sparse.append({'scenario':scenario,'feature':col,'positive_training_days':int(X.loc[valid,col].sum())})
  fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
  for ax,name in zip(axes,['LightGBM + history','XGBoost + history']):
   values=pd.DataFrame(rows);g=values.loc[(values.scenario==scenario)&(values.model==name)].nlargest(12,'importance').sort_values('importance')
   ax.barh(g.feature,g.importance);ax.set(title=scenario+' '+name,xlabel='Training split importance (association)')
  fig.savefig(out/f'{scenario}_importance.png',dpi=150);plt.show();plt.close(fig)
 pd.DataFrame(rows).to_csv(out/'feature_importance.csv',index=False);pd.DataFrame(sparse).to_csv(out/'holiday_support.csv',index=False)
 return pd.DataFrame(rows),pd.DataFrame(sparse)
