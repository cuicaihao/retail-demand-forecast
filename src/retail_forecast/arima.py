from .common import *
from scipy.linalg import qr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
NAMES=['ARIMA','SARIMA weekly','Business regression','Regression + ARIMA','Seasonal naive']
def fit_arima(history,scenario,name,window):
 exog=name in ['Business regression','Regression + ARIMA'];scaler=None;cols=[];selected=[];X=None
 if exog:
  features=make_features(history,scenario);cols=list(features.columns[features.loc[valid_labels(history)].std()>1e-8])
  scaler=StandardScaler().fit(features.loc[valid_labels(history),cols]);a=scaler.transform(features[cols])
  _,r,pivot=qr(np.column_stack([np.ones(len(a)),a])[valid_labels(history)],pivoting=True,mode='economic')
  rank=int((abs(np.diag(r))>1e-8).sum());selected=sorted(int(j-1) for j in pivot[:rank] if j>0);X=a[:,selected]
 order=(0,0,0) if name=='Business regression' else (1,0,1) if exog else (1,1,1)
 seasonal=(1,0,0,7) if name=='SARIMA weekly' else (0,0,0,0)
 y=masked_y(history).to_numpy();model=SARIMAX(y,exog=X,order=order,seasonal_order=seasonal,trend='c',enforce_stationarity=True,enforce_invertibility=True)
 fitted=model.fit(disp=False,maxiter=200)
 if not fitted.mle_retvals.get('converged',False):fitted=model.fit(start_params=fitted.params,disp=False,maxiter=600)
 bundle={'result':fitted,'scaler':scaler,'columns':cols,'selected':selected,'exog':X,'y':y}
 return bundle,{'order':str(order),'seasonal_order':str(seasonal),'converged':bool(fitted.mle_retvals.get('converged',False)),'AIC':float(fitted.aic)}
def arima_exog(bundle,frame,scenario):
 return None if bundle['scaler'] is None else bundle['scaler'].transform(make_features(frame,scenario)[bundle['columns']])[:,bundle['selected']]
def predict_arima(bundle,history,future,scenario,name):
 result=bundle['result'];end=len(history)
 if end<len(bundle['y']):result=result.apply(bundle['y'][:end],exog=None if bundle['exog'] is None else bundle['exog'][:end],refit=False)
 return np.asarray(result.get_forecast(len(future),exog=arima_exog(bundle,future,scenario)).predicted_mean)
def diagnose_arima(finals):
 out=RUNS_DIR/'arima';rows=[];coefficients=[]
 for scenario,frame in datasets.items():
  bundle=finals[(scenario,'Regression + ARIMA')];r=bundle['result']
  params=dict(zip(r.param_names,r.params))
  for name,value in params.items():coefficients.append({'scenario':scenario,'parameter':name,'value':float(value)})
  for j,param_name in enumerate(r.model.exog_names):
   idx=bundle['selected'][j];feature=bundle['columns'][idx];beta=params[param_name]
   coefficients.append({'scenario':scenario,'parameter':param_name,'feature':feature,'value':float(beta),'original_unit_slope':float(beta/bundle['scaler'].scale_[idx]),'training_mean':float(bundle['scaler'].mean_[idx])})
  burn=max(r.loglikelihood_burn,28);innov=r.filter_results.standardized_forecasts_error[0][burn:]
  usable=valid_labels(frame.iloc[:TRAIN_END+1]).to_numpy()[burn:]
  indices=np.flatnonzero(usable & np.isfinite(innov));runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
  longest=max(runs,key=len);innov=innov[longest]
  assert len(innov)>28
  lb=acorr_ljungbox(innov,lags=[7,14],model_df=2,return_df=True)
  for lag,row in lb.iterrows():rows.append({'scenario':scenario,'lag':lag,**row.to_dict()})
  fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
  axes[0].plot(innov);axes[0].set(title=scenario+' standardized innovations (longest contiguous valid segment)',ylabel='Standardized innovation')
  plot_acf(innov,lags=28,ax=axes[1]);axes[1].set(title='Training diagnostic: consecutive daily observations only')
  fig.savefig(out/f'{scenario}_innovations.png',dpi=150);plt.show();plt.close(fig)
  future=frame.iloc[TRAIN_END+1:];forecast=r.get_forecast(HORIZON,exog=arima_exog(bundle,future,scenario));ci=np.asarray(forecast.conf_int(alpha=.2))
  fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True);ax.plot(future.ds,future.y,'ko-',ms=3,label='Sales');ax.plot(future.ds,forecast.predicted_mean,label='Raw forecast')
  ax.fill_between(future.ds,ci[:,0],ci[:,1],alpha=.2,label='Nominal 80% interval (raw)');ax.set(title=scenario+' final test interval');ax.legend();ax.tick_params(axis='x',rotation=20)
  fig.savefig(out/f'{scenario}_interval.png',dpi=150);plt.show();plt.close(fig)
  mask=valid_labels(future).to_numpy();rows.append({'scenario':scenario,'diagnostic':'test_interval','coverage':float(((future.y.to_numpy()[mask]>=ci[mask,0])&(future.y.to_numpy()[mask]<=ci[mask,1])).mean()),'mean_width':float((ci[mask,1]-ci[mask,0]).mean())})
  r.save(out/f'{scenario}_dynamic_regression.pkl')
  import joblib
  joblib.dump(bundle,out/f'{scenario}_dynamic_bundle.joblib')
  restored=joblib.load(out/f'{scenario}_dynamic_bundle.joblib')
  np.testing.assert_allclose(predict_arima(restored,frame.iloc[:TRAIN_END+1],future,scenario,'Regression + ARIMA'),forecast.predicted_mean,rtol=1e-8,atol=1e-8)
 pd.DataFrame(coefficients).to_csv(out/'parameters.csv',index=False);pd.DataFrame(rows).to_csv(out/'diagnostics.csv',index=False)
 return pd.DataFrame(coefficients),pd.DataFrame(rows)
