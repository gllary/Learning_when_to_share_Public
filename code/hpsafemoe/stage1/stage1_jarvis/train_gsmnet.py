"""Official GSMNet/Ignite training engine with full/OOF membership and safe export."""
import argparse
import json
import os
import time
import importlib.metadata
import numpy as np
from common import ROOT, atomic_json, registry, training_labels, validation_labels, output_dir, protocol_hash, sha256

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--fold', type=int, default=-1)
    p.add_argument('--batch', type=int)
    a = p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torch_geometric.loader import DataLoader
    from ignite.engine import Events
    from strong_backend import load_graph_dataset
    from strong_reference import gsm_configuration
    from gsm_reference import training_function
    from train_strong_expert import LabeledView
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    assert torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(0.78)
    row = next((r for r in registry()['tasks'] if r['id'] == a.task))
    assert row['model_id'] == 'gsmnet_model'
    out = output_dir(a.task, a.fold)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    if (out / 'official_run/gsmnet_mbj_bandgap_history_val.json').exists():
        raise RuntimeError('Interrupted GSMNet engine: inspect official checkpoints before resuming; no silent restart')
    ids = np.load(ROOT / 'data/universe_ids.npy').astype(str)
    index = {j: i for i, j in enumerate(ids)}
    labels = training_labels(a.task, a.fold)
    val = validation_labels(a.task)
    ds = load_graph_dataset('mbj_bandgap')
    config = gsm_configuration()
    reference = config.dict()
    config.output_dir = str(out / 'official_run')
    config.write_predictions = False
    config.progress = False
    config.log_tensorboard = False
    if a.batch is not None and a.batch != config.batch_size:
        raise RuntimeError('Unapproved batch override')
    options = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)
    train = DataLoader(LabeledView(ds, [index[j] for j in labels], list(labels.values())), shuffle=True, **options)
    validation = DataLoader(LabeledView(ds, [index[j] for j in val], list(val.values())), shuffle=False, **options)
    assert len(train) > 0 and len(validation) > 0
    function, namespace, body_hash = training_function()
    capture = []
    constructed = []
    native_constructor = namespace['GSMNet']
    native_trainer = namespace['create_supervised_trainer']
    started = time.time()

    def constructor(cfg):
        model = native_constructor(cfg)
        model.fc_out.register_forward_pre_hook(lambda module, args: (capture.clear(), capture.append(args[0])) and None)
        constructed.append(model)
        return model

    def trainer_factory(*args, **kwargs):
        engine = native_trainer(*args, **kwargs)

        @engine.on(Events.EPOCH_COMPLETED)
        def progress(e):
            atomic_json(out / 'progress.json', dict(epoch=e.state.epoch, epochs=config.epochs, seconds=time.time() - started, status='training', engine='official_Ignite'))
        return engine
    namespace['GSMNet'] = constructor
    namespace['create_supervised_trainer'] = trainer_factory
    model, history = function(config.dict(), train_val_test_loaders=[train, validation, None])
    assert constructed == [model]
    score = float(history['validation']['mae'][-1])
    assert np.isfinite(score)
    metadata = dict(task=a.task, model_id=row['model_id'], fold=a.fold, trained_ids=list(labels), validation_ids=list(val), test_labels_read=False, official_validation_labels_read=True, protocol_sha256=protocol_hash(), retrained_from_scratch=True, reference_config=reference, effective_config=config.dict(), official_training_body_sha256=body_hash, checkpoint_rule='official_default_final_epoch', packages={n: importlib.metadata.version(n) for n in ['torch', 'torch-geometric', 'numpy', 'jarvis-tools', 'pytorch-ignite', 'mamba-ssm']})
    atomic_json(out / 'config.json', metadata)
    atomic_json(out / 'validation.json', dict(score=score, metric='MAE', selected_epoch=config.epochs, checkpoint_rule='official_default_final_epoch', test_labels_read=False))
    model.eval()

    def forward(batch):
        capture.clear()
        pred = model(batch.to('cuda')).reshape(-1)
        assert len(capture) == 1
        return (pred, capture[0])
    with torch.inference_mode():
        checkpoint = out / 'model.pt'
        torch.save(model.state_dict(), checkpoint)
        predictions = np.lib.format.open_memmap(out / 'prediction.npy', mode='w+', dtype='float32', shape=(len(ids),))
        features = np.lib.format.open_memmap(out / 'hidden.npy', mode='w+', dtype='float32', shape=(len(ids), 256))
        for i, batch in enumerate(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)):
            pred, hidden = forward(batch)
            predictions[i] = pred.item()
            features[i] = hidden[0].cpu().numpy()
        predictions.flush()
        features.flush()
        assert np.isfinite(predictions).all() and np.isfinite(features).all()
    atomic_json(out / 'complete.json', dict(task=a.task, fold=a.fold, rows=len(ids), hidden_dim=256, checkpoint_sha256=sha256(checkpoint), test_labels_read=False, seconds=time.time() - started))
    print('EXPERT_COMPLETE', a.task, a.fold, flush=True)
if __name__ == '__main__':
    main()
