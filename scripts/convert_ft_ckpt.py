import argparse
import configparser
from datetime import datetime
import multiprocessing
import os
from pathlib import Path
from shutil import copyfile

import numpy as np

def split_ckpt(hidden_units, in_dir, out_dir, filename, factor):
    if (filename.find("args.txt") != -1
        or filename.find("config.ini") != -1
        or filename.find("model.wpe.bin") != -1
        or filename.find("model.wte.bin") != -1
        or filename.find("input_layernorm.weight") != -1
        or filename.find("input_layernorm.bias") != -1
        or filename.find("attention.dense.bias") != -1
        or filename.find("post_attention_layernorm.weight") != -1
        or filename.find("post_attention_layernorm.bias") != -1
        or filename.find("mlp.dense_4h_to_h.bias") != -1
        or filename.find("final_layernorm.weight") != -1
        or filename.find("final_layernorm.bias") != -1):
        
        copyfile(in_dir / filename, out_dir / filename)
        return

    orig_gpu_id = int(filename.split(".")[-2])

    file_prefix = ".".join(filename.split(".")[:-2])
    val = np.fromfile(f"{in_dir}/{filename}", dtype=np.float16)
    if filename.find("attention.dense.weight") != -1 or filename.find("mlp.dense_4h_to_h.weight") != -1:
        vals = np.split(val.reshape(-1, hidden_units), factor, axis=0)
    elif filename.find("attention.query_key_value.weight") != -1:
        val = val.reshape(hidden_units, -1)
        qkv = np.split(val, 3, axis=-1)
        q_vals = np.split(qkv[0], factor, axis=-1)
        k_vals = np.split(qkv[1], factor, axis=-1)
        v_vals = np.split(qkv[2], factor, axis=-1)
        vals = []
        for k in range(factor):
            vals.append(np.concatenate([q_vals[k], k_vals[k], v_vals[k]], axis=-1))
    elif filename.find("attention.query_key_value.bias") != -1:
        val = val.reshape(3, -1)
        q_vals = np.split(val[0], factor, axis=-1)
        k_vals = np.split(val[1], factor, axis=-1)
        v_vals = np.split(val[2], factor, axis=-1)
        vals = []
        for k in range(factor):
            vals.append(np.concatenate([q_vals[k], k_vals[k], v_vals[k]], axis=0))
    elif filename.find("mlp.dense_h_to_4h.weight") != -1:
        vals = np.split(val.reshape(hidden_units, -1), factor, axis=-1)
    elif filename.find("mlp.dense_h_to_4h.bias") != -1:
        vals = np.split(val, factor, axis=-1)
    else:
        print(f"[ERROR] cannot find key '{filename}'")
        return

    for k in range(factor):
        vals[k].tofile(f"{out_dir}/{file_prefix}.{orig_gpu_id * factor + k}.bin")

def merge_ckpt(hidden_units, in_dir, out_dir, filename, factor):
    if (filename.find("args.txt") != -1
        or filename.find("config.ini") != -1
        or filename.find("model.wpe.bin") != -1
        or filename.find("model.wte.bin") != -1
        or filename.find("input_layernorm.weight") != -1
        or filename.find("input_layernorm.bias") != -1
        or filename.find("attention.dense.bias") != -1
        or filename.find("post_attention_layernorm.weight") != -1
        or filename.find("post_attention_layernorm.bias") != -1
        or filename.find("mlp.dense_4h_to_h.bias") != -1
        or filename.find("final_layernorm.weight") != -1
        or filename.find("final_layernorm.bias") != -1):
        
        copyfile(in_dir / filename, out_dir / filename)
        return

    orig_gpu_id = int(filename.split(".")[-2])
    if (orig_gpu_id % factor != 0):
        return

    file_prefix = ".".join(filename.split(".")[:-2])
    merge_files = []
    vals = []
    new_file = f"{out_dir}/{file_prefix}.{orig_gpu_id // factor}.bin"
    for k in range(factor):
        gpu_id = orig_gpu_id + k
        vals.append(np.fromfile(f"{in_dir}/{file_prefix}.{gpu_id}.bin", dtype=np.float16))
        merge_files.append(f"{in_dir}/{file_prefix}.{gpu_id}.bin")

    if filename.find("attention.dense.weight") != -1 or filename.find("mlp.dense_4h_to_h.weight") != -1:
        vals = [val.reshape(-1, hidden_units) for val in vals]
        np.concatenate(vals, axis=0).tofile(f"{out_dir}/{file_prefix}.{orig_gpu_id // factor}.bin")
    elif filename.find("attention.query_key_value.weight") != -1:
        qkv_vals = [np.split(val.reshape(hidden_units, -1), 3, axis=-1) for val in vals]
        q_val = np.concatenate([qkv_vals[k][0] for k in range(factor)], axis=-1)
        k_val = np.concatenate([qkv_vals[k][1] for k in range(factor)], axis=-1)
        v_val = np.concatenate([qkv_vals[k][2] for k in range(factor)], axis=-1)
        np.concatenate([q_val, k_val, v_val], axis=-1).tofile(f"{out_dir}/{file_prefix}.{orig_gpu_id // factor}.bin")
    elif filename.find("attention.query_key_value.bias") != -1:
        qkv_vals = [np.split(val.reshape(3, -1), 3, axis=0) for val in vals]
        q_val = np.concatenate([qkv_vals[k][0] for k in range(factor)], axis=-1)
        k_val = np.concatenate([qkv_vals[k][1] for k in range(factor)], axis=-1)
        v_val = np.concatenate([qkv_vals[k][2] for k in range(factor)], axis=-1)
        np.concatenate([q_val, k_val, v_val], axis=0).tofile(f"{out_dir}/{file_prefix}.{orig_gpu_id // factor}.bin")
    elif filename.find("mlp.dense_h_to_4h.weight") != -1:
        vals = [val.reshape(hidden_units, -1) for val in vals]
        np.concatenate(vals, axis=-1).tofile(f"{out_dir}/{file_prefix}.{orig_gpu_id // factor}.bin")
    elif filename.find("mlp.dense_h_to_4h.bias") != -1:
        np.concatenate(vals, axis=-1).tofile(f"{out_dir}/{file_prefix}.{orig_gpu_id // factor}.bin")
    else:
        print(f"[ERROR] cannot find key '{filename}'")

def convert_checkpoint(args):
    out_dir = Path(args.out_dir) / f"{args.out_gpu_num:d}-gpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(args.in_dir) / f"{args.in_gpu_num:d}-gpu"
   
    if args.in_gpu_num > args.out_gpu_num:
        assert args.in_gpu_num % args.out_gpu_num == 0
        is_merge_ckpt = True
        factor = int(args.in_gpu_num / args.out_gpu_num)
    else:
        assert args.out_gpu_num % args.in_gpu_num == 0
        is_merge_ckpt = False
        factor = int(args.out_gpu_num / args.in_gpu_num)

    hidden_units = 5120
    pool = multiprocessing.Pool(args.processes)
    pool.starmap(
            merge_ckpt if is_merge_ckpt == True else split_ckpt,
            [
               (hidden_units, in_dir, out_dir, filename, factor) for filename in os.listdir(in_dir)
            ],
    )
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-out_dir", "-o", type=str, help="directory name of output checkpoint", required=True)
    parser.add_argument("-in_dir", "-i", type=str, help="directory name of input checkpoint", required=True)
    parser.add_argument("-in_gpu_num", "-i_g", type=int, help="The number of gpus in input checkpoint", required=True)
    parser.add_argument("-out_gpu_num", "-o_g", type=int, help="The number of gpus in output checkpoint", required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 64)", default=64)
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    convert_checkpoint(args)
