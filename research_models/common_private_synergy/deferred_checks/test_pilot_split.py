import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from research_models.common_private_synergy import patient_pilot as cli
from research_models.common_private_synergy import pilot_split as split
from slotspe_proof import patient as legacy


@pytest.fixture
def invented(tmp_path):
    meta=tmp_path/'metadata';features=tmp_path/'features';features.mkdir()
    for folder in ('clinical/all','raw_rna_data_inter','signatures','splits/5fold','splits/5fold_uni2h'):
        (meta/folder).mkdir(parents=True)
    train=[f'T{i}' for i in range(6)]+['BAD_CLIN','BAD_SLIDE']
    val=['V0','V1','NO_RNA']
    def csv_split(name,tr,va):
        pd.DataFrame({'train':pd.Series(tr),'val':pd.Series(va)}).to_csv(meta/'splits'/name/'blca/fold_0.csv',index=False)
    (meta/'splits/5fold/blca').mkdir();(meta/'splits/5fold_uni2h/blca').mkdir()
    csv_split('5fold_uni2h',train,val)
    # Same counts, different membership and partitions: counts alone cannot
    # reveal the historical hardcoded-directory bug.
    csv_split('5fold',['V0']+train[1:],['T0','V1','NO_RNA'])
    rows=[]
    for i,pid in enumerate(train+val):
        rows.append({'case id':pid,'survival_months_dss':np.nan if pid=='BAD_CLIN' else float(i+1),
                     'censorship_dss':0.,'wsi':'s'+pid+'.svs'})
        if pid not in ('BAD_CLIN','BAD_SLIDE','NO_RNA'):
            torch.save(torch.arange(3*(i+1),dtype=torch.float32).reshape(i+1,3),features/('s'+pid+'.pt'))
    pd.DataFrame(rows).to_csv(meta/'clinical/all/blca.csv',index=False)
    pd.DataFrame({p:[1.,2.,3.] for p in train+val if p!='NO_RNA'},index=['a','b','c']).to_csv(meta/'raw_rna_data_inter/blca_rna_inter.csv')
    pd.DataFrame({'path1':['a','b'],'path2':['c',None]}).to_csv(meta/'signatures/combine_signatures.csv',index=False)
    cfg=legacy._config({'source_root':str(cli.ROOT/'third_party/SlotSPE'),'metadata_root':str(meta),
        'features_root':str(features),'study':'blca','label_col':'survival_months_dss','signature':'combine',
        'folds':[0],'seeds':[3],'patches':4096,'splits_dir':'5fold_uni2h'})
    return cfg,train,val


@pytest.fixture
def prepared(invented,tmp_path):
    cfg,tr,va=invented;root=tmp_path/'prepared_run'
    manifest=split.prepare(cfg,root)
    cli.write(root/'resolved_config.json',cfg)
    return cfg,root,manifest


def test_actual_requested_split_and_complete_exclusions(prepared):
    cfg,root,m=prepared
    row=m['folds']['0']
    assert row['train']==[f'T{i}' for i in range(6)]
    assert row['val']==['V0','V1']
    assert {(e['patient_id'],e['reason']) for e in row['excluded']}=={
        ('BAD_CLIN','missing_clinical_endpoint'),('BAD_SLIDE','at_least_one_missing_slide_feature'),
        ('NO_RNA','missing_rna_patient_column')}
    verified=split.validate_split(cfg,m)
    assert verified['folds']['0']['groups']['train']['requested']['count']==8
    assert verified['folds']['0']['groups']['train']['retained']['count']==6
    # A one-patch patient is retained with exactly one real patch, not dropped
    # for being below the 4096 budget and not cached as 4096 fake real patches.
    cache=torch.load(root/m['patients']['T0']['cache_path'],weights_only=True)
    assert cache['patches'].shape==(1,3)
    assert legacy._manifest(cfg,root,verify_cache=True)==m


def test_native_class_unchanged_and_never_selects_for_pilot(invented,monkeypatch,tmp_path):
    cfg,_,_=invented
    native=legacy._dataset_module(cfg);before=native.SurvivalDataset._load_split
    class Native:
        def _load_split(self):raise AssertionError('native hardcoded selector must never execute')
    observations={};Wrapped=split.split_dataset_class(Native,cfg,observations)
    ds=Wrapped();ds.fold=0;ds.split_key='train'
    class Factory:clinical_df=pd.DataFrame({'case id':['T0','V0']})
    ds.dataset_factory=Factory()
    assert ds._load_split()['case id'].tolist()==['T0']
    assert native.SurvivalDataset._load_split is before


@pytest.mark.parametrize('mutation', ['missing_identity','false_actual_hash','wrong_member_same_count',
    'swap_train_val','missing_exclusion','duplicate_exclusion','no_reason','wrong_split_name',
    'missing_input_hash','bad_counts','wrong_order','overlap'])
def test_bad_manifest_refused(prepared,mutation):
    cfg,_,original=prepared;m=copy.deepcopy(original);row=m['folds']['0']
    if mutation=='missing_identity':del m['split_identity']
    elif mutation=='false_actual_hash':m['split_identity']['folds']['0']['actual']['sha256']='0'*64
    elif mutation=='wrong_member_same_count':row['train'][0]='V0'
    elif mutation=='swap_train_val':row['train'],row['val']=row['val'],row['train']
    elif mutation=='missing_exclusion':row['excluded'].pop()
    elif mutation=='duplicate_exclusion':row['excluded'].append(dict(row['excluded'][0]))
    elif mutation=='no_reason':row['excluded'][0].pop('reason')
    elif mutation=='wrong_split_name':row['split_csv']=row['split_csv'].replace('5fold_uni2h','5fold')
    elif mutation=='missing_input_hash':m['input_csv_sha256'].pop(row['split_csv'])
    elif mutation=='bad_counts':row['counts']['train']+=1
    elif mutation=='wrong_order':row['train'].reverse()
    elif mutation=='overlap':row['val'].append(row['train'][0])
    with pytest.raises(ValueError):split.validate_split(cfg,m)


def test_split_contents_changed_after_preparation(prepared):
    cfg,_,m=prepared
    path=Path(m['folds']['0']['split_csv'])
    text=path.read_text(encoding='utf-8');path.write_text(text.replace('T0','NEW'),encoding='utf-8')
    with pytest.raises(ValueError,match='identity changed'):split.validate_split(cfg,m)


@pytest.mark.parametrize('bad', ['duplicate','overlap','space'])
def test_requested_csv_invalid(invented,bad):
    cfg,tr,va=invented
    if bad=='duplicate':tr[1]=tr[0]
    elif bad=='overlap':va[0]=tr[0]
    else:tr[0]=' T0'
    path=Path(cfg['metadata_root'])/'splits/5fold_uni2h/blca/fold_0.csv'
    pd.DataFrame({'train':pd.Series(tr),'val':pd.Series(va)}).to_csv(path,index=False)
    with pytest.raises(ValueError):split.read_split(cfg,0)


def test_symlink_alias_to_identical_requested_file_allowed(invented,tmp_path):
    cfg,_,_=invented
    original=Path(cfg['metadata_root'])/'splits/5fold_uni2h/blca/fold_0.csv'
    target=tmp_path/'canonical_fold.csv';original.replace(target)
    try:original.symlink_to(target)
    except OSError as error:pytest.skip('host lacks symlink permission: '+str(error))
    info,_=split.read_split(cfg,0)
    assert info['realpath']==str(target.resolve())
    m=split.prepare(cfg,tmp_path/'symlink_cache')
    assert split.validate_split(cfg,m)['identity_sha256']
    other=tmp_path/'other_fold.csv';other.write_text(target.read_text().replace('T0','WRONG'))
    original.unlink();original.symlink_to(other)
    with pytest.raises(ValueError):split.validate_split(cfg,m)


def test_cli_legacy_admission_fails_before_cache_verification_or_engine(prepared,monkeypatch,tmp_path):
    cfg,root,m=prepared;m.pop('split_identity')
    cli.write(root/'prepared/manifest.json',m)
    def forbidden(*a,**k):pytest.fail('invalid split reached legacy cache verification')
    monkeypatch.setattr(legacy,'_manifest',forbidden)
    out=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='legacy/unverified'):
        cli.main(['--run','--prepared-root',str(root),'--output-root',str(out)])
    assert not out.exists()


def test_check_prepared_is_metadata_only_and_does_not_modify(prepared,monkeypatch,capsys):
    cfg,root,m=prepared
    before=split.file_hash(root/'prepared/manifest.json')
    def forbidden(*a,**k):pytest.fail('split metadata check loaded a tensor')
    monkeypatch.setattr(torch,'load',forbidden)
    cli.main(['--check-prepared','--prepared-root',str(root)])
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='split_metadata_verified' and not report['model_executed']
    assert split.file_hash(root/'prepared/manifest.json')==before


def test_boundary_rule_matches_left_closed_cut():
    from research_models.common_private_synergy.patient_pilot_engine import label_times
    edges=np.array([0.,1.,2.,3.]);times=np.array([.5,1.,2.,2.5])
    cuts=pd.cut(times,edges,labels=False,right=False)
    labels=label_times(torch.tensor(times),torch.ones(4,dtype=torch.bool),torch.tensor(edges[1:-1]))
    np.testing.assert_array_equal(cuts,labels.numpy())
    assert np.searchsorted(edges[1:-1],times,side='left').tolist()!=cuts.tolist()


def test_prepare_refuses_existing_artifacts(prepared):
    cfg,root,m=prepared;before=split.file_hash(root/'prepared/manifest.json')
    with pytest.raises(FileExistsError):split.prepare(cfg,root)
    assert split.file_hash(root/'prepared/manifest.json')==before


def test_cli_preparation_uses_explicit_wrapper_not_legacy(invented,tmp_path,monkeypatch):
    cfg,_,_=invented
    def forbidden(*a,**k):pytest.fail('CLI invoked legacy hardcoded preparation')
    monkeypatch.setattr(legacy,'prepare',forbidden)
    config_path=tmp_path/'data_config.json';cli.write(config_path,cfg)
    out=tmp_path/'cli_prepare'
    cli.main(['--prepare','--data-config',str(config_path),'--output-root',str(out)])
    m=json.loads((out/'prepared/manifest.json').read_text(encoding='utf-8'))
    assert m['folds']['0']['train']==[f'T{i}' for i in range(6)]
    assert split.validate_split(cfg,m)['identity_sha256']


def test_engine_without_admission_rejected_before_any_tensor_work(monkeypatch,tmp_path):
    from research_models.common_private_synergy.patient_pilot_engine import run_prepared
    monkeypatch.setattr(torch,'set_num_threads',lambda *a:pytest.fail('invalid admission reached tensor work'))
    with pytest.raises(ValueError,match='verified split admission'):
        run_prepared(tmp_path,tmp_path/'out',{}, {}, {})
    assert not (tmp_path/'out').exists()


def test_changed_admission_stops_before_validation(tmp_path,monkeypatch):
    from research_models.common_private_synergy import patient_pilot_engine as engine
    original=engine.run_prepared;calls=[]
    def identity():
        calls.append(1)
        return {'identity_sha256':'first' if len(calls)==1 else 'changed'}
    def run(*a,**k):
        k['split_guard']=identity
        return original(*a,**k)
    monkeypatch.setattr(engine,'run_prepared',run)
    torch.set_num_threads(1)
    out=tmp_path/'changed_audit'
    with pytest.raises(RuntimeError,match='before evaluation'):engine.synthetic_smoke(out)
    assert len(calls)==2
    assert not (out/'run/validation_predictions.json').exists()
