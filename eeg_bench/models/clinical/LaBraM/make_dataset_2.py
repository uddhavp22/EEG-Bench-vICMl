from tqdm import tqdm
import os
from multiprocessing import Pool, Process, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from .utils_2 import writer_task, process_one_abnormal, process_one_cli_unm, process_one_epilepsy, process_one_multilabel, LaBraMDataset2, NeuroGPTDataset2, get_channels
from ....config import get_config_value
from ....utils.utils import get_multilabel_tasks
import h5py
import logging

def make_dataset(X, y, meta, task_name, model_name, chunk_len_s, is_train, use_cache, **kwargs):
    # Create or override the HDF5 file.
    h5_folder = os.path.join(get_config_value("data"), "make_dataset")
    split_key = "train" if is_train else "test"
    h5_path = os.path.join(
        h5_folder,
        f"{task_name}_{model_name}_{meta[0]['name'].replace(' ', '_')}_{chunk_len_s}_{split_key}_{sum(len(obj) for obj in X)}.h5",
    )
    if os.path.exists(h5_path) and use_cache:
        # Validate h5 file is not corrupted before trusting cache
        try:
            with h5py.File(h5_path, 'r') as hf:
                if '/recordings' not in hf or len(hf['/recordings']) == 0:
                    raise ValueError("Empty or missing /recordings group")
        except Exception as e:
            print(f"[Warning] Cached h5 is corrupted ({e}). Deleting and rebuilding: {h5_path}")
            os.remove(h5_path)
        else:
            print(f"[Info] Dataset already exists at {h5_path}. Loading existing dataset.")
            if model_name == "NeuroGPTModel":
                return NeuroGPTDataset2(h5_path, is_train, get_channels(task_name), **kwargs)
            else:
                return LaBraMDataset2(h5_path, is_train, get_channels(task_name))

    if not os.path.exists(h5_folder):
        os.makedirs(h5_folder)
    with h5py.File(h5_path, 'w') as hf:
        hf.create_group('/recordings')

    manager = Manager()
    output_queue = manager.Queue()
    writer = Process(target=writer_task, args=(output_queue, h5_path))
    writer.start()
    n_jobs = 10 #os.cpu_count() - 1 #ALERT! MAJOR CHANGE!!
    if n_jobs < 1:
        n_jobs = 1

    
    if "abnormal" in task_name:
        X = X[0]
        if y is None:
            y = [None] * len(X)
        else:
            y = y[0]
        t_channels = get_channels(task_name)
        parameters = [(i, raw, label, model_name, chunk_len_s) for i, (raw, label) in enumerate(zip(X, y))]
        worker_func = partial(process_one_abnormal, output_queue=output_queue)
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(worker_func, p): p for p in parameters}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing abnormal data"):
                p = futures[future]
                try:
                    future.result()
                except Exception as e:
                    idx, raw, *_ = p
                    fname = getattr(raw, "filenames", None)
                    raise RuntimeError(f"Failed at idx={idx}, filenames={fname}, err={e}") from e

    elif "epilepsy" in task_name:
        X, montage_types = X[0], meta[0]["montage_type"]
        if y is None:
            y = [None] * len(X)
        else:
            y = y[0]
        t_channels = get_channels(task_name)
        parameters = [(i, raw, label, montage, task_name, model_name, chunk_len_s) for i, (raw, label, montage) in enumerate(zip(X, y, montage_types))]
        worker_func = partial(process_one_epilepsy, output_queue=output_queue)
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(worker_func, p): p for p in parameters}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing epilepsy data"):
                p = futures[future]
                try:
                    future.result()
                except Exception as e:
                    idx, raw, *_ = p
                    fname = getattr(raw, "filenames", None)
                    raise RuntimeError(f"Failed at idx={idx}, filenames={fname}, err={e}") from e
        logging.info("--------- All recordings have been processed.")
    
    elif task_name in get_multilabel_tasks():
        if y is None:
            y = [None] * len(X)
        t_channels = get_channels(task_name)
        last_idx = 0
        for data, labels, m in zip(X, y, meta):
            if labels is None:
                labels = [None] * len(data)
            dataset_name = m["name"]
            parameters = [(i + last_idx, raw, label, t_channels, model_name, chunk_len_s) for i, (raw, label) in enumerate(zip(data, labels))]
            worker_func = partial(process_one_multilabel, output_queue=output_queue)
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(worker_func, p): p for p in parameters}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing multilabel data"):
                    p = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        idx, raw, *_ = p
                        fname = getattr(raw, "filenames", None)
                        raise RuntimeError(f"Failed at idx={idx}, filenames={fname}, err={e}") from e
            last_idx += len(data)
            logging.info(f"--------- All recordings from {dataset_name} have been processed.")
        logging.info("--------- All recordings have been processed.")
    
    else:
        if y is None:
            y = [None] * len(X)
        last_idx = 0
        for data, labels, m in zip(X, y, meta):
            if labels is None:
                labels = [None] * len(data)
            sfreq = m['sampling_frequency']
            dataset_name = m['name']
            o_channels = m['channel_names']
            o_channels = [ch.upper() for ch in o_channels]
            t_channels = get_channels(task_name)
            parameters = [(i + last_idx, signals, label, o_channels, sfreq, model_name, task_name, chunk_len_s) for i, (signals, label) in enumerate(zip(data, labels))]
            worker_func = partial(process_one_cli_unm, output_queue=output_queue)
            n_jobs = n_jobs // 2
            if n_jobs < 1:
                n_jobs = 1
            with Pool(1) as pool:
                list(tqdm(pool.imap(worker_func, parameters), total=len(parameters),
                        desc=f"Processing {dataset_name}"))
            last_idx += len(data)
            logging.info(f"--------- All recordings from {dataset_name} have been processed.")
        logging.info("--------- All recordings have been processed.")

    # Signal the writer process that all workers are done.
    print("[Main] Signaling writer process that all workers are done.")
    output_queue.put(None)
    writer.join()

    if model_name == "NeuroGPTModel":
        return NeuroGPTDataset2(h5_path, is_train, t_channels, **kwargs)
    else:
        return LaBraMDataset2(h5_path, is_train, t_channels)
