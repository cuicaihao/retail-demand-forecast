from .common import *
import copy,torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from .tft import fit_tft,predict_tft,restore_tft
NAMES=['MLP','LSTM','TFT','Seasonal naive']
NEURAL_MAX_EPOCHS=35;NEURAL_BATCH_SIZE=64;WINDOW_STRIDE=7
TFT_MODE='train' # Historical checkpoints are opt-in because their manifests use an older data snapshot.
torch.set_num_threads(CPU_THREADS)
class WindowNetwork(nn.Module):
 def __init__(self,kind,features,hidden=32):
  super().__init__();self.kind=kind
  self.encoder=nn.Sequential(nn.Linear(ENCODER_LENGTH,hidden),nn.ReLU()) if kind=='MLP' else nn.LSTM(features+1,hidden,batch_first=True)
  self.head=nn.Sequential(nn.Linear(hidden+features,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,past,future):
  if self.kind=='MLP':context=self.encoder(past[:,:,0])
  else:_,(h,_)=self.encoder(past);context=h[-1]
  context=context[:,None,:].expand(-1,future.shape[1],-1)
  return self.head(torch.cat([context,future],dim=-1)).squeeze(-1)

def preprocessor(history,scenario):
 y=history.loc[valid_labels(history),'y'];mean=float(y.mean());std=max(float(y.std()),1.)
 return {'mean':mean,'std':std,'scaler':StandardScaler().fit(make_features(history,scenario))}
def inputs(history,future,scenario,prep):
 y=masked_y(history).ffill().fillna(prep['mean']).tail(ENCODER_LENGTH).to_numpy(float)
 assert len(y)==ENCODER_LENGTH
 p=prep['scaler'].transform(make_features(history.tail(ENCODER_LENGTH),scenario))
 f=prep['scaler'].transform(make_features(future,scenario))
 return np.column_stack([(y-prep['mean'])/prep['std'],p]).astype('float32'),f.astype('float32')
def training_tensors(history,scenario,prep):
 past=[];future=[];targets=[];weights=[]
 for origin in range(ENCODER_LENGTH-1,len(history)-HORIZON,WINDOW_STRIDE):
  h=history.iloc[:origin+1];f=history.iloc[origin+1:origin+1+HORIZON]
  a,b=inputs(h,f,scenario,prep);past.append(a);future.append(b)
  targets.append(((f.y.to_numpy(float)-prep['mean'])/prep['std']).astype('float32'))
  weights.append(valid_labels(f).to_numpy(dtype='float32'))
 return tuple(torch.from_numpy(np.stack(a)) for a in [past,future,targets,weights])
def masked_mse(p,y,w):return (((p-y)**2)*w).sum()/w.sum().clamp_min(1.)
def train_epochs(model,tensors,epochs,validation=None):
 loader=DataLoader(TensorDataset(*tensors),batch_size=NEURAL_BATCH_SIZE,shuffle=True,generator=torch.Generator().manual_seed(SEED))
 opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
 logs=[];best=float('inf');best_epoch=1;stale=0
 for epoch in range(1,epochs+1):
  model.train();losses=[]
  for p,f,y,w in loader:
   opt.zero_grad();loss=masked_mse(model(p,f),y,w);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  row={'epoch':epoch,'train_loss':float(np.mean(losses))}
  if validation is not None:
   model.eval()
   with torch.no_grad():v=float(masked_mse(model(validation[0],validation[1]),validation[2],validation[3]))
   row['validation_loss']=v
   if v<best:best=v;best_epoch=epoch;stale=0
   else:stale+=1
  logs.append(row)
  if validation is not None and stale>=8:break
 return best_epoch,logs

def fit_neural(history,scenario,name,window):
 if name=='TFT':
  if TFT_MODE=='checkpoint':return restore_tft(history,scenario,window['regime'])
  bundle,details=fit_tft(history,scenario,train_end=len(history)-1,checkpoint_tag=f'{scenario}_{window["regime"]}');details['mode']='retrained';return bundle,details
 core=history.iloc[:-HORIZON];v=history.iloc[-HORIZON:];prep=preprocessor(core,scenario)
 train=training_tensors(core,scenario,prep);a,b=inputs(core,v,scenario,prep)
 y=((v.y.to_numpy(float)-prep['mean'])/prep['std']).astype('float32');w=valid_labels(v).to_numpy(dtype='float32')
 validation=tuple(torch.from_numpy(x)[None] for x in [a,b,y,w])
 torch.manual_seed(SEED);candidate=WindowNetwork(name,len(feature_names[scenario]));epochs,selection=train_epochs(candidate,train,NEURAL_MAX_EPOCHS,validation)
 prep=preprocessor(history,scenario);torch.manual_seed(SEED);model=WindowNetwork(name,len(feature_names[scenario]))
 _,final_log=train_epochs(model,training_tensors(history,scenario,prep),epochs);model.eval()
 out=RUNS_DIR/'neural';out.mkdir(parents=True,exist_ok=True);tag=f'{scenario}_{window["regime"]}_{name}'
 pd.DataFrame(selection).to_csv(out/f'{tag}_selection.csv',index=False)
 torch.save({'state_dict':model.state_dict(),'kind':name,'feature_count':len(feature_names[scenario]),'mean':prep['mean'],'std':prep['std'],'scaler_mean':prep['scaler'].mean_.tolist(),'scaler_scale':prep['scaler'].scale_.tolist(),'feature_names':feature_names[scenario]},out/f'{tag}.pt')
 return {'model':model,'prep':prep},{'selected_epochs':epochs,'mode':'new_training','parameters':sum(p.numel() for p in model.parameters()),'training_windows':len(train[0]),'validation_MSE_scaled':min(r['validation_loss'] for r in selection)}

def predict_neural(bundle,history,future,scenario,name):
 if name=='TFT':
  f=pd.concat([history,future]);origin=len(history)-1
  raw=predict_tft(bundle,f,scenario,origin);changed=predict_tft(bundle,f,scenario,origin,placeholder=12345.)
  np.testing.assert_allclose(raw,changed,rtol=1e-6,atol=1e-5);return raw
 a,b=inputs(history,future,scenario,bundle['prep']);changed=future.copy();changed['y']=12345.
 ca,cb=inputs(history,changed,scenario,bundle['prep']);np.testing.assert_array_equal(a,ca);np.testing.assert_array_equal(b,cb)
 with torch.no_grad():raw=bundle['model'](torch.from_numpy(a)[None],torch.from_numpy(b)[None]).numpy().reshape(-1)
 return raw*bundle['prep']['std']+bundle['prep']['mean']

def neural_curves():
 out=RUNS_DIR/'neural';fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
 for ax,scenario in zip(axes,SCENARIOS):
  for kind in ['MLP','LSTM']:
   g=pd.read_csv(out/f'{scenario}_final_{kind}_selection.csv');ax.plot(g.epoch,np.sqrt(g.validation_loss),label=kind+' validation')
  ax.set(title=scenario,xlabel='Epoch',ylabel='Scaled validation RMSE');ax.legend()
 fig.savefig(out/'learning_curves.png',dpi=150);plt.show();plt.close(fig)
