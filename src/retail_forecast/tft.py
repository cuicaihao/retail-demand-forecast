from .common import *
import torch,lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback,EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer,TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE
TFT_MAX_EPOCHS=25;TFT_BATCH_SIZE=128;TFT_HIDDEN_SIZE=16
OUTPUT_DIR=RUNS_DIR/'neural'
torch.set_num_threads(CPU_THREADS)
def progress(message):print(message,flush=True)
class EpochChoice(Callback):
    def __init__(self,label):
        self.label=label;self.best_loss=float("inf");self.best_epoch=1
    def on_validation_epoch_end(self,trainer,model):
        if trainer.sanity_checking:return
        value=float(trainer.callback_metrics["val_loss"])
        if value < self.best_loss:self.best_loss=value;self.best_epoch=trainer.current_epoch+1
        if (trainer.current_epoch+1)%5==0:
            progress(f"{self.label}: epoch {trainer.current_epoch+1}, val_loss={value:.4f}")

class TrainProgress(Callback):
    def __init__(self,label):self.label=label
    def on_train_epoch_end(self,trainer,model):
        if (trainer.current_epoch+1)%5==0:
            progress(f"{self.label}: epoch {trainer.current_epoch+1}")

def tft_frame(frame, scenario, mean, std):
    out=make_features(frame,scenario).reset_index(drop=True)
    out["time_idx"]=frame.time_idx.to_numpy(dtype=int)
    out["series"]=scenario
    y=masked_y(frame).ffill().fillna(mean).to_numpy()
    out["target_z"]=(y-mean)/std
    out["label_weight"]=valid_labels(frame).astype(float).to_numpy()
    return out

def training_dataset(frame,scenario):
    usable=frame.loc[valid_labels(frame),"y"]
    mean=float(usable.mean());std=max(float(usable.std()),1.0)
    prepared=tft_frame(frame,scenario,mean,std)
    dataset=TimeSeriesDataSet(prepared,time_idx="time_idx",target="target_z",group_ids=["series"],
              weight="label_weight",min_encoder_length=ENCODER_LENGTH,max_encoder_length=ENCODER_LENGTH,
              min_prediction_length=HORIZON,max_prediction_length=HORIZON,
              static_categoricals=["series"],time_varying_known_reals=feature_names[scenario],
              time_varying_unknown_reals=["target_z"],target_normalizer=None,
              add_relative_time_idx=True,randomize_length=False)
    return dataset,mean,std

def tft_inference_dataset(bundle, full_frame, scenario, origin, placeholder=0.0):
    assert origin+HORIZON < len(full_frame)
    visible=full_frame.iloc[:origin+HORIZON+1].copy()
    # Hide all target labels and diagnostic truth after the origin before preprocessing.
    visible.loc[visible.time_idx > origin,"y"]=np.nan
    prepared=tft_frame(visible,scenario,bundle["mean"],bundle["std"])
    decoder=prepared.time_idx > origin
    prepared.loc[decoder,"target_z"]=placeholder
    prepared.loc[decoder,"label_weight"]=0.0
    return TimeSeriesDataSet.from_dataset(bundle["dataset"],prepared,predict=True,stop_randomization=True)

def new_tft(dataset):
    return TemporalFusionTransformer.from_dataset(dataset,learning_rate=.005,
               hidden_size=TFT_HIDDEN_SIZE,attention_head_size=2,hidden_continuous_size=8,
               lstm_layers=1,dropout=.1,output_size=1,loss=RMSE(),
               reduce_on_plateau_patience=5,log_interval=-1)

def trainer(epochs,callbacks=()):
    return pl.Trainer(accelerator="cpu",devices=1,max_epochs=epochs,logger=False,
               enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
               callbacks=list(callbacks),gradient_clip_val=.1,num_sanity_val_steps=0,
               deterministic=True,default_root_dir=str(OUTPUT_DIR / "lightning"))

def fit_tft(full_frame,scenario,train_end=None,checkpoint_tag=None):
    if train_end is None: train_end=TRAIN_END
    inner_end=train_end-HORIZON
    if checkpoint_tag is None: checkpoint_tag=scenario
    history=full_frame.iloc[:train_end+1].copy()
    core=history.iloc[:inner_end+1].copy()
    core_dataset,mean,std=training_dataset(core,scenario)
    selection_std=std
    # Validation labels are used for the loss only; they are not decoder input variables.
    validation_prepared=tft_frame(history,scenario,mean,std)
    validation_dataset=TimeSeriesDataSet.from_dataset(core_dataset,validation_prepared,
                                                      predict=True,stop_randomization=True)
    pl.seed_everything(SEED,workers=True)
    selection=new_tft(core_dataset)
    choice=EpochChoice(scenario+" selection")
    train_loader=core_dataset.to_dataloader(train=True,batch_size=TFT_BATCH_SIZE,num_workers=0)
    validation_loader=validation_dataset.to_dataloader(train=False,batch_size=1,num_workers=0)
    selection_trainer=trainer(TFT_MAX_EPOCHS,[choice,EarlyStopping(monitor="val_loss",patience=6,mode="min")])
    selection_trainer.fit(selection,train_dataloaders=train_loader,val_dataloaders=validation_loader)
    epochs=choice.best_epoch
    dataset,mean,std=training_dataset(history,scenario)
    pl.seed_everything(SEED,workers=True)
    final=new_tft(dataset)
    final_trainer=trainer(epochs,[TrainProgress(scenario+" final")])
    final_trainer.fit(final,train_dataloaders=dataset.to_dataloader(
         train=True,batch_size=TFT_BATCH_SIZE,num_workers=0))
    final.eval()
    final_trainer.save_checkpoint(OUTPUT_DIR / f"{checkpoint_tag}_tft.ckpt")
    return {"model":final,"dataset":dataset,"mean":mean,"std":std}, {
       "selected_epochs":epochs,"validation_RMSE_original":float(choice.best_loss*selection_std),
       "encoder_length":ENCODER_LENGTH,"hidden_size":TFT_HIDDEN_SIZE,
       "target_mean":mean,"target_std":std,"validation_target_std":selection_std}

def predict_tft(bundle,full_frame,scenario,origin,placeholder=0.0):
    dataset=tft_inference_dataset(bundle,full_frame,scenario,origin,placeholder)
    loader=dataset.to_dataloader(train=False,batch_size=1,num_workers=0)
    raw=bundle["model"].predict(loader,mode="prediction",trainer_kwargs={
         "accelerator":"cpu","devices":1,"logger":False,"enable_progress_bar":False,
         "enable_model_summary":False,"enable_checkpointing":False})
    if hasattr(raw,"output"):raw=raw.output
    values=raw.detach().cpu().numpy().reshape(-1)*bundle["std"]+bundle["mean"]
    assert len(values)==HORIZON
    return values

def restore_tft(history,scenario,regime):
    checkpoint=ROOT/'models'/'reference'/'tft'/f'{scenario+"_"+regime if regime!="final" else scenario}_tft.ckpt'
    checkpoint_dir=ROOT/'models'/'reference'/'tft'
    manifest=json.loads((checkpoint_dir/'comparison_manifest.json').read_text())
    for name,digest in manifest['input_sha256'].items():
        assert hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest()==digest
    dataset,mean,std=training_dataset(history,scenario)
    key=scenario+'_'+regime
    saved=manifest['tft_target_scaling'][scenario] if regime=='final' else manifest['rolling_tft_target_scaling'][key]
    np.testing.assert_allclose([mean,std],[saved['mean'],saved['std']],rtol=1e-10)
    model=new_tft(dataset)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict'],strict=True)
    model.eval()
    table=pd.read_csv(checkpoint_dir/('fit_details.csv' if regime=='final' else 'rolling_fit_details.csv'))
    row=table.loc[table.scenario.eq(scenario)&table.model.eq('TFT')]
    if regime!='final':row=row.loc[row.regime.eq(regime)]
    row=row.iloc[0]
    return {'model':model,'dataset':dataset,'mean':mean,'std':std}, {'mode':'verified_checkpoint','recorded_training_seconds':float(row.fit_seconds),'selected_epochs':int(row.selected_epochs)}
