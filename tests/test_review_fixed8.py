"""CPU regressions for the review; no model/SSD performance assertions."""
from __future__ import annotations
import json
from types import SimpleNamespace

import numpy as np
import pytest

from contiguous_fuxian.promixed import select_promixed_gqa_blocks, selection_experiment, current_selection_experiment
from contiguous_fuxian.group_coverage import select_group_blocks, coverage_of
from contiguous_fuxian.quantized_key_index import (
    INDEX_FORMAT, QuantizedKeyIndex, pack_signed_int4, unpack_signed_int4,
    quantize_symmetric_int4, dequantize_int4_numpy,
)
from contiguous_fuxian.audited_reprefill import checked_summary, token_hash, validate_prefix_identity, main


def disagreement_rows():
    return [[9, 8, 0, 0, 0, 0, 0, 0], [0, 0, 9, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 9, 8, 0, 0], [0, 0, 0, 0, 0, 0, 9, 8]]


def test_fixed8_really_overrides_adaptive_period_and_restores_context():
    rows = disagreement_rows()
    assert select_promixed_gqa_blocks(rows, keep_blocks=2).period == 4
    with selection_experiment(fixed_period=8):
        assert select_promixed_gqa_blocks(rows, keep_blocks=2).period == 8
        with selection_experiment(fixed_period=2):
            assert select_promixed_gqa_blocks(rows, keep_blocks=2).period == 2
        assert current_selection_experiment()['fixed_period'] == 8
    assert current_selection_experiment() is None


def test_context_reset_on_error():
    with pytest.raises(RuntimeError):
        with selection_experiment():
            raise RuntimeError('test')
    assert current_selection_experiment() is None


def test_explicit_no_coverage_not_zero_fraction():
    # First group reserves block 0, but global normalized mean favors block 1.
    rows = [[9, 8, 0], [0, 9, 8], [0, 9, 8], [0, 9, 8]]
    legacy = select_promixed_gqa_blocks(rows, keep_blocks=1, coverage_fraction=0)
    assert legacy.selected_blocks == (0,)
    with selection_experiment(coverage_enabled=False):
        disabled = select_promixed_gqa_blocks(rows, keep_blocks=1,
            utility_max_weight=0, utility_mean_weight=1, utility_vote_weight=0)
    assert disabled.selected_blocks == (1,)


@pytest.mark.parametrize('budget',[True, 1.2, float('nan'), None])
def test_bad_budget_rejected(budget):
    with pytest.raises(ValueError):
        select_promixed_gqa_blocks([[1, 2]], keep_blocks=budget)


@pytest.mark.parametrize('strategy',['balanced','mean','normalized_mean','max'])
@pytest.mark.parametrize('k',[1,2,8])
def test_group_strategies_exact_budget(strategy,k):
    result = select_group_blocks(disagreement_rows(),keep_blocks=k,strategy=strategy)
    assert len(set(result.priority_blocks)) == k
    assert all(0 <= x <= 1+1e-12 for x in result.retained_mass)
    assert all(0 <= x <= 1+1e-12 for x in result.relative_to_topk)


def test_balanced_selection_order_is_group_permutation_invariant():
    rng=np.random.default_rng(22); a=rng.random((4,13))
    x=select_group_blocks(a,keep_blocks=5)
    y=select_group_blocks(a[[3,1,0,2]],keep_blocks=5)
    assert x.priority_blocks == y.priority_blocks


def test_balanced_uses_coverage_not_fixed_group_quota():
    a=np.array([[10,9,8,0],[0,0,0,10]],dtype=float)
    result=select_group_blocks(a,keep_blocks=2)
    assert 3 in result.priority_blocks
    assert set(result.priority_blocks) == {0,3}


def test_zero_scores_and_partial_zero_groups():
    result=select_group_blocks(np.zeros((4,8)),keep_blocks=3)
    assert result.priority_blocks == (0,1,2)
    assert result.relative_to_topk == (0,0,0,0)
    assert select_group_blocks([[0,0,0],[1,9,0]],keep_blocks=1).priority_blocks == (1,)


@pytest.mark.parametrize('bad',[[[float('nan'),1]], [[-1,1]], [], [[1],[2,3]]])
def test_bad_group_scores(bad):
    with pytest.raises(ValueError):
        select_group_blocks(bad,keep_blocks=1)


def test_trace_is_opt_in_and_names_invocations():
    trace=[]
    with selection_experiment(strategy='balanced',trace=trace):
        decision=select_promixed_gqa_blocks(disagreement_rows(),keep_blocks=2)
    assert len(trace)==1 and trace[0]['invocation']==0
    assert trace[0]['period']==8
    assert trace[0]['selected_blocks']==list(decision.selected_blocks)


@pytest.mark.parametrize('bad',[np.empty((1,0),np.int8),np.array([[1.5,2.0]]),np.array([[np.nan,0]])])
def test_int4_invalid_pack_input(bad):
    with pytest.raises(ValueError):
        pack_signed_int4(bad)


@pytest.mark.parametrize('bad',[np.array(1),np.array([[-1]]),np.array([[256]]),np.array([[1.5]])])
def test_int4_invalid_packed_bytes(bad):
    with pytest.raises(ValueError):
        unpack_signed_int4(bad,width=2)


def test_all_int4_codes_roundtrip():
    values=np.arange(-7,8,dtype=np.int8); values=np.r_[values,0][None]
    assert np.array_equal(unpack_signed_int4(pack_signed_int4(values),width=16),values)


def test_float_group_size_rejected():
    with pytest.raises(ValueError):
        quantize_symmetric_int4(np.ones((1,1,4)),group_size=2.5)


def write_index(tmp_path,layers=10):
    d=tmp_path/'task';d.mkdir()
    codes,scales=quantize_symmetric_int4(np.array([[[1.,-1.,.5,-.5]]]),group_size=2)
    for layer in range(layers):
        codes.tofile(d/f'layer_{layer:02d}.codes.u8')
        scales.tofile(d/f'layer_{layer:02d}.scales.f16')
    payload={'format':INDEX_FORMAT,'bits':4,'selector_kv_head_ids':[0],
        'tasks':{'task':{'directory':'task','prefix_tokens':1,'layers':layers,
            'kv_heads':1,'head_dim':4,'group_size':2}}}
    (tmp_path/'manifest.json').write_text(json.dumps(payload))
    return codes.nbytes+scales.nbytes


def test_anchor_preload_byte_accounting_and_guard(tmp_path):
    pytest.importorskip('torch')
    per_layer=write_index(tmp_path,28)
    with selection_experiment(preload_anchors_only=True):
        index=QuantizedKeyIndex(tmp_path)
    assert index.preloaded_layer_ids('task')==(0,8,16,24)
    assert not index.task_is_preloaded('task')
    assert index.preload_task('task')==4*per_layer
    assert index.task_is_preloaded('task')
    assert index.preload_task('task')==0
    _,n,source=index.load_layer('task',8,device='cpu')
    assert source=='cpu' and n==per_layer
    with pytest.raises(RuntimeError,match='non-anchor'):
        index.load_layer('task',1,device='cpu')


def test_historical_preload_still_all_layers(tmp_path):
    pytest.importorskip('torch'); per_layer=write_index(tmp_path)
    index=QuantizedKeyIndex(tmp_path)
    assert index.preload_task('task')==10*per_layer
    assert index.preloaded_layer_ids('task')==tuple(range(10))


@pytest.mark.parametrize('bad',[float('nan'),-1.0,float('inf')])
def test_bad_scale_reference(bad):
    with pytest.raises(ValueError):
        dequantize_int4_numpy(np.array([[[0]]],np.uint8),np.array([[[bad]]]),head_dim=2)


def summary_template():
    return {'tasks':{'trec':{'mean_promixed_period':8}},'runtime':{
        'runtime_variant':'hyperinfer+promixed-gqa-adaptive-p8+k4-selector-index'}}


def test_summary_rejects_fake_fixed8_label():
    s=summary_template();s['tasks']['trec']['mean_promixed_period']=5.1
    with pytest.raises(RuntimeError):
        checked_summary(s,strategy='legacy',coverage_enabled=True,anchor_preload=True,trace_enabled=False)


def test_summary_records_actual_options():
    s=checked_summary(summary_template(),strategy='balanced',coverage_enabled=False,
        anchor_preload=True,trace_enabled=True)
    assert '+promixed-gqa-fixed-p8' in s['runtime']['runtime_variant']
    assert s['runtime']['latency_contains_selector_trace']
    assert s['runtime']['actual_payload_quantization'].startswith('none')


def test_prefix_identity_validation():
    tok=lambda text,**kw: SimpleNamespace(input_ids=[ord(c) for c in text])
    metadata={'t':{'prefix_tokens':2,'token_hash':token_hash([65,66])}}
    validate_prefix_identity([{'task':'t','prefix_text':'AB'}],tok,metadata)
    with pytest.raises(ValueError,match='token hash'):
        validate_prefix_identity([{'task':'t','prefix_text':'AC'}],tok,metadata)


def test_dry_run_does_not_create_output_or_import_transformers(tmp_path):
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps(dict(model_type='qwen2',num_hidden_layers=28,
        num_key_value_heads=4,num_attention_heads=28)))
    store=tmp_path/'store'/'trec';store.mkdir(parents=True)
    (store/'metadata.json').write_text(json.dumps(dict(layers=28,kv_heads=4,head_dim=128,prefix_tokens=2,token_hash='stub')))
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'trec.jsonl').write_text(json.dumps(dict(task='trec',prefix_text='AB'))+'\n')
    output=tmp_path/'new-run'
    argv=['--model-path',str(model),'--bundle-dir',str(bundle),'--store-root',str(store.parent),
        '--flexgen-root',str(tmp_path),'--flexgen-kv-dir',str(tmp_path),
        '--output-dir',str(output),'--store-tasks','trec','--dry-run']
    assert main(argv)==0
    assert not output.exists()


@pytest.mark.parametrize('dtype_name',['float16','bfloat16','float32'])
def test_prefix_writer_obeys_bf16_byte_contract(tmp_path,dtype_name):
    # Execute exact repository function ASTs, without importing GPU-only runner
    # dependencies. The test still exercises the actual writer/reader bodies.
    import ast
    from pathlib import Path
    from dataclasses import dataclass
    import os
    torch=pytest.importorskip('torch')
    path=Path(__file__).resolve().parents[1]/'src/contiguous_fuxian/sparse_qwen_reprefill.py'
    tree=ast.parse(path.read_text())
    names={'PrefixStoreInfo','contiguous_spans','_write_prefix_tensor','_read_layer_chunks'}
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    body += [node for node in tree.body if getattr(node,'name',None) in names]
    scope=dict(dataclass=dataclass,Path=Path,os=os,BYTES_PER_BFLOAT16=2)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),scope)
    values=torch.tensor([[[[1.,-1.],[2.,-2.],[3.,-3.]]]],dtype=getattr(torch,dtype_name))
    stored=tmp_path/'values.bf16'
    scope['_write_prefix_tensor'](stored,values)
    info=scope['PrefixStoreInfo'](task='t',prefix_tokens=3,kv_heads=1,head_dim=2,layers=1,token_hash='unused')
    restored=scope['_read_layer_chunks'](stored,info,2,[0,1])
    assert torch.equal(restored,values.to(torch.bfloat16))
    assert stored.stat().st_size == values.numel()*2
