import argparse
import logging
import os
import pathlib
import pprint
import threading
import time
import torch
from torch.multiprocessing import Process, Queue, set_start_method, set_sharing_strategy
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm.auto import tqdm
from multiqueue_worker import multiqueue_worker
from utils_nystrom import humanize_units, init_torch, init_logging, load_model, num_classes_of, save_ntk, log_memory
from datasets import get_VD, get_TD


local = threading.local()

def wrap_loader(loader):
    batch_start = 0
    for batch in loader:
        data = batch[0] if isinstance(batch, list) else batch
        batch_len = data.size(0)
        batch_stop = batch_start + batch_len
        yield (data, (slice(batch_start, batch_stop), batch_len))
        batch_start = batch_stop

def _init_compute_gradients(model, params_slice, buffer_size):
    if "model" not in local.__dict__:
        local.model = model.cuda()

    if "grad" not in local.__dict__:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grad_size = param_count + (-param_count % buffer_size[1])
        local.grad = torch.zeros(grad_size, dtype=torch.float, device="cuda")

    slot_start = 0
    local.active_params = []
    for param, local_param in zip(model.parameters(), local.model.parameters()):
        if not param.requires_grad:
            continue
        local_param.requires_grad = False
        slot_stop = slot_start + param.numel()
        if not (slot_stop <= params_slice.start or slot_start >= params_slice.stop):
            local.active_params.append((slice(slot_start, slot_stop), local_param))
            local_param.requires_grad = True
        slot_start = slot_stop
    local.params_slice = params_slice

def _compute_gradients(model, params_slice, buffer_size, data, batch_info):
    if not hasattr(local, "params_slice") or local.params_slice!= params_slice:
        _init_compute_gradients(model, params_slice, buffer_size)
    local.grad.zero_()
    gpu_buffer = torch.zeros(buffer_size, device="cuda")
    data = data.cuda()
    for i in range(data.size(0)):
        local.model.zero_grad(set_to_none=True)
        local.model.forward(data[i : i + 1])[0].backward()

        for slot, param in local.active_params:
            local.grad[slot.start:slot.stop].copy_(param.grad.flatten())
        gpu_buffer[i] = local.grad[local.params_slice]
    del data
    torch.cuda.empty_cache()
    return gpu_buffer, batch_info

def compute_gradients(in_queue, out_queue, model, params_slice, loader, out_buffer):
    buffer_size = (loader.batch_size, out_buffer.size(1))
    in_flight = 0
    pbar = tqdm(total=len(loader.dataset), desc="Computing Gradients")

    for data, batch_info in wrap_loader(loader):
        args = (model, params_slice, buffer_size, data.clone(), batch_info)
        in_queue.put((_compute_gradients, args))
        in_flight += 1

        if in_flight >= 36:
            gpu_buffer, (batch_slice, batch_len) = out_queue.get()
            out_buffer[batch_slice].copy_(gpu_buffer[:batch_len])
            del gpu_buffer
            in_flight -= 1
            pbar.update(batch_len)

    while in_flight > 0:
        gpu_buffer, (batch_slice, batch_len) = out_queue.get()
        out_buffer[batch_slice].copy_(gpu_buffer[:batch_len])
        del gpu_buffer
        in_flight -= 1
        pbar.update(batch_len)

    pbar.close()


def _compute_XYt(X_chunk, Y_chunk, buffer_size, buffer_dtype):
    if "buffer" not in local.__dict__ or local.buffer.shape != buffer_size:
        local.buffer = torch.zeros(buffer_size, dtype=buffer_dtype, device="cuda")
    
    X_chunk = X_chunk.to(local.buffer)
    Y_chunk = Y_chunk.to(local.buffer)
    local.buffer.addmm_(X_chunk, Y_chunk.T)

# def _return_XYt_buffer():
#     assert "buffer" in local.__dict__
#     return local.buffer
def _return_XYt_buffer():
    if "buffer" not in local.__dict__:

        return None
    return local.buffer

# def _clear_XYt_buffer():
#     assert "buffer" in local.__dict__
#     local.buffer.zero_()
def _clear_XYt_buffer():
  
    if "buffer" in local.__dict__:
        del local.buffer
    torch.cuda.empty_cache()

def compute_XYt(

    in_queues_devices, out_queue, X, Y, out, row_chunksize, col_chunksize
):
    """
    Computes out = X @ Y.T in a distributed, chunked fashion.
    """
    in_flight = 0
    pbar = tqdm(total=X.size(0), desc="Assembling Kernel")
    
    num_matmul_workers = len(in_queues_devices)
    task_idx = 0

    for i in range(0, X.size(0), row_chunksize):
        row_slice = slice(i, i + row_chunksize)
        
        for j in range(0, X.size(1), col_chunksize):
            col_slice = slice(j, j + col_chunksize)
            
            X_chunk = X[row_slice, col_slice].clone()
            Y_chunk = Y[:, col_slice].clone()

            current_buffer_shape = out[row_slice].shape
            args = (X_chunk, Y_chunk, current_buffer_shape, out.dtype)
            
            #  round-robin fashion
            dispatch_to_queue = in_queues_devices[task_idx % num_matmul_workers]
            dispatch_to_queue.put((_compute_XYt, args))
            task_idx += 1
            # ---
            
            in_flight += 1

            if in_flight >= 3 * num_matmul_workers:
                _ = out_queue.get()
                in_flight -= 1
        while in_flight > 0:
            _ = out_queue.get()
            in_flight -= 1

        for in_queue in in_queues_devices:
            in_queue.put((_return_XYt_buffer, ()))
            in_flight += 1

        # while in_flight > 0:
        #     gpu_buffer = out_queue.get()
        #     out[row_slice].add_(gpu_buffer.cpu())
        #     in_flight -= 1
        while in_flight > 0:
            gpu_buffer = out_queue.get()
            # checking the buffer is not None before adding
            if gpu_buffer is not None:
                out[row_slice].add_(gpu_buffer.cpu())
            in_flight -= 1

        for in_queue in in_queues_devices:
            in_queue.put((_clear_XYt_buffer, ()))
            in_flight += 1
        
        while in_flight > 0:
            _ = out_queue.get()
            in_flight -= 1
        
        pbar.update(X[row_slice].size(0))
        
    pbar.close()


def compute_nystrom_ntk(
    model,
    train_loader,
    valid_loader,  
    test_loader,
    posX_loader,
    landmark_loader,
    num_devices=None,
    workers_per_device=1,
    grad_chunksize=None,
    mm_col_chunksize=None,
    mm_row_chunksize=None,
    pin_memory=True,
    ntk_dtype=torch.double,
    init_torch_kwargs={},
):
    if num_devices is None:
        num_devices = torch.cuda.device_count()
    if grad_chunksize is None or mm_col_chunksize is None or mm_row_chunksize is None:
        raise ValueError("grad_chunksize, mm_col_chunksize, and mm_row_chunksize must be specified.")

    logging.info(f"Executing on {num_devices} device(s) with {workers_per_device} workers per device.")

    num_workers = num_devices * workers_per_device
    in_queue_grad = Queue()
    in_queues_devices = [Queue() for _ in range(num_devices)]
    out_queue = Queue()

    worker_processes = []
    for i in range(num_workers):
        device = i % num_devices
        i_in_queues = [in_queue_grad]
        if i < num_devices:
            i_in_queues.append(in_queues_devices[i])
        args = (device, init_torch_kwargs, i_in_queues, out_queue)
        p = Process(target=multiqueue_worker, args=args)
        p.start()
        worker_processes.append(p)

    model.zero_grad(set_to_none=True)
    model.eval()

    n_train = len(train_loader.dataset)
    n_valid = len(valid_loader.dataset)
    n_test = len(test_loader.dataset)
    n_landmarks = len(landmark_loader.dataset)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_batches = (param_count + grad_chunksize - 1) // grad_chunksize

    log_memory("Allocating landmark and chunk buffers")
    grads_landmarks = torch.zeros((n_landmarks, grad_chunksize), dtype=torch.float, pin_memory=pin_memory)

    grads_data_chunk = torch.zeros((mm_row_chunksize, grad_chunksize), dtype=torch.float, pin_memory=pin_memory)


    ntk_train_landmarks = torch.zeros((n_train, n_landmarks), dtype=ntk_dtype)
    ntk_valid_landmarks = torch.zeros((n_valid, n_landmarks), dtype=ntk_dtype)
    ntk_test_landmarks = torch.zeros((n_test, n_landmarks), dtype=ntk_dtype)
    ntk_aug_landmarks = torch.zeros((n_train, n_landmarks), dtype=ntk_dtype)
    ntk_landmarks_landmarks = torch.zeros((n_landmarks, n_landmarks), dtype=ntk_dtype)

    logging.info(f"Train: {n_train}, Valid: {n_valid}, Test: {n_test}, Landmarks: {n_landmarks}")
    logging.info(f"Total parameters: {param_count}, splitting into {param_batches} chunks of size {grad_chunksize}.")


    for i, params_start in enumerate(range(0, param_count, grad_chunksize)):
        logging.info(f"--- Starting parameter chunk {i + 1}/{param_batches} ---")
        params_stop = min(params_start + grad_chunksize, param_count)
        params_slice = slice(params_start, params_stop)
        current_chunk_size = params_stop - params_start

        grads_landmarks_view = grads_landmarks[:, :current_chunk_size]
        
      
        logging.info("Computing gradients for landmark points...")
        compute_gradients(in_queue_grad, out_queue, model, params_slice, landmark_loader, grads_landmarks_view)
        torch.cuda.empty_cache()

  
        datasets_to_process = {
            "train": (train_loader, ntk_train_landmarks),
            "valid": (valid_loader, ntk_valid_landmarks),
            "test": (test_loader, ntk_test_landmarks),
            "aug": (posX_loader, ntk_aug_landmarks),
        }
        

        for name, (loader, ntk_out) in datasets_to_process.items():
            logging.info(f"Processing '{name}' dataset...")
            n_samples = len(loader.dataset)
            for row_start in range(0, n_samples, mm_row_chunksize):
                row_stop = min(row_start + mm_row_chunksize, n_samples)
                current_row_slice = slice(row_start, row_stop)
                current_row_count = row_stop - row_start
                
                data_subset = torch.utils.data.Subset(loader.dataset, range(row_start, row_stop))
                data_chunk_loader = DataLoader(data_subset, batch_size=loader.batch_size, num_workers=loader.num_workers)

                grads_data_chunk_view = grads_data_chunk[:current_row_count, :current_chunk_size]
                
             
                compute_gradients(in_queue_grad, out_queue, model, params_slice, data_chunk_loader, grads_data_chunk_view)
                
               
                compute_XYt(
                    in_queues_devices, out_queue,
                    grads_data_chunk_view, grads_landmarks_view,
                    ntk_out[current_row_slice],
                    mm_row_chunksize, mm_col_chunksize
                )
            
      
        logging.info("Assembling partial kernel K_mm...")
        compute_XYt(
            in_queues_devices, out_queue,
            grads_landmarks_view, grads_landmarks_view,
            ntk_landmarks_landmarks, mm_row_chunksize, mm_col_chunksize
        )

        logging.info(f"Finished parameter chunk {i + 1}/{param_batches}")
        torch.cuda.empty_cache()


    for _ in range(num_workers):
        in_queue_grad.put(None)
    for p in worker_processes:
        p.join()

    return ntk_train_landmarks, ntk_valid_landmarks, ntk_test_landmarks, ntk_landmarks_landmarks, ntk_aug_landmarks

if __name__ == "__main__":
    
    set_start_method("spawn", force=True)
    set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Compute Nystrom-approximated NTK with distributed matrix multiplication.")
    parser.add_argument("--dataset", type=str, default='adult', help="Name of the dataset.")
    parser.add_argument("--model", type=str, default='resmlp', help="Name of the model architecture.")

    parser.add_argument("--in-features", type=int, default=110, help="Input features for ResMLP.")
    parser.add_argument("--mlp-width", type=int, default=1032, help="Width of ResMLP layers.")
    parser.add_argument("--num-blocks", type=int, default=3, help="Number of residual blocks in ResMLP.")
    parser.add_argument("--train_path", type=str, default='/income.data')
    parser.add_argument("--test_path", type=str, default='/income.test')

    parser.add_argument("--savedir", type=str, default="./ntks", help="Directory to save the computed NTKs.")
    parser.add_argument("--logdir", type=str, default="./my_logs", help="Directory to save logs.")
    parser.add_argument("--workers-per-device", type=int, default=4)
    parser.add_argument("--grad-chunksize", type=int, default=4000000, help="Size of parameter chunks for gradient calculation.")
    parser.add_argument("--mm-col-chunksize", type=int, default=500000, help="Column chunk size for matrix multiplication.")
    parser.add_argument("--mm-row-chunksize", type=int, default=8000, help="Row chunk size for matrix multiplication.")
    parser.add_argument("--ntk-dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--loader-batch-size", type=int, default=128)
    parser.add_argument("--loader-num-workers", type=int, default=4)
    parser.add_argument("--no-pinned-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--n_landmark", type=int, default=2000, help="Number of landmark points.")
    
    args = parser.parse_args()

    init_logging(f"nystrom_dist_{args.dataset}", args.logdir)
    logging.info(f"args =\n{pprint.pformat(vars(args))}")

    init_torch_kwargs = {
        "allow_tf32": args.allow_tf32,
        "benchmark": args.benchmark,
        "deterministic": args.deterministic,
    }
    init_torch(**init_torch_kwargs, verbose=True)
    

    logging.info("Loading data and landmarks...")
    class Config:
        def __init__(self):
            self.dataset = args.dataset
            self.n_landmark = args.n_landmark
            self.batch_size = args.loader_batch_size
            self.loader_workers = args.loader_num_workers
            self.device = 'cpu'
            self.kernel_fn = True
            self.loss_f = 'bt'
            self.classes = num_classes_of(args.dataset)
            self.Ny_centers = args.n_landmark
            self.model = args.model
            self.train_path = args.train_path
            self.test_path = args.test_path

    cnf = Config()

    if args.dataset in ['cifar10', 'imagenet']:
        loaders = get_VD(cnf, args.savedir)
    elif args.dataset in ['adult', 'covtype', 'higgs']:
        loaders = get_TD(cnf, args.savedir)
    else:
        raise ValueError(f"Unknown dataset type for {args.dataset}")

    logging.info(
    f"Data loaded: {len(loaders['train'].dataset)} train, "
    f"{len(loaders['valid'].dataset)} validation, "
    f"{len(loaders['test'].dataset)} test, "
    f"{len(loaders['landmark'].dataset)} landmarks,"
    f"{len(loaders['aug'].dataset)} posX."
)

    model_kwargs = {
        "in_features": args.in_features,
        "mlp_width": args.mlp_width,
        "num_blocks": args.num_blocks,
    }
    model = load_model(args.model, **model_kwargs)
    logging.info(f"Loaded model '{args.model}'.")

    kwargs = {
        "workers_per_device": args.workers_per_device,
        "grad_chunksize": args.grad_chunksize,
        "mm_col_chunksize": args.mm_col_chunksize,
        "mm_row_chunksize": args.mm_row_chunksize,
        "pin_memory": args.pin_memory,
        "init_torch_kwargs": init_torch_kwargs,
        "ntk_dtype": torch.float32 if args.ntk_dtype == "float32" else torch.float64,
    }
    
    K_nm, K_vm, K_tm, K_mm, K_pnm = compute_nystrom_ntk(
        model,
        loaders["train"],
        loaders["valid"],
        loaders["test"],
        loaders["aug"],
        loaders["landmark"],
        **kwargs
    )

    # --- Saving Results ---
    logging.info(f"Successfully computed Nystrom NTKs.")
    logging.info(f"K_nm (train) size: {K_nm.size()}")
    logging.info(f"K_pnm (posX) size: {K_pnm.size()}")
    logging.info(f"K_vm (valid) size: {K_vm.size()}")
    logging.info(f"K_tm (test) size: {K_tm.size()}")
    logging.info(f"K_mm (landmark) size: {K_mm.size()}")
    
    assert K_nm.shape == (len(loaders["train"].dataset), len(loaders["landmark"].dataset))
    assert K_pnm.shape == (len(loaders["aug"].dataset), len(loaders["landmark"].dataset))
    assert K_vm.shape == (len(loaders["valid"].dataset), len(loaders["landmark"].dataset))
    assert K_tm.shape == (len(loaders["test"].dataset), len(loaders["landmark"].dataset))
    assert K_mm.shape == (len(loaders["landmark"].dataset), len(loaders["landmark"].dataset))

    save_ntk(K_nm, args.savedir, f"{args.dataset}_{args.model}_K_nm")
    save_ntk(K_pnm, args.savedir, f"{args.dataset}_{args.model}_K_pnm")
    save_ntk(K_vm, args.savedir, f"{args.dataset}_{args.model}_K_vm")
    save_ntk(K_tm, args.savedir, f"{args.dataset}_{args.model}_K_tm")
    save_ntk(K_mm, args.savedir, f"{args.dataset}_{args.model}_K_mm")
    
    logging.info("All tasks completed.")