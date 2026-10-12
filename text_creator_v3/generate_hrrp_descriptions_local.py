import argparse
import base64
import io
import json
import os
import re
import time
import traceback
import multiprocessing as mp
from datetime import datetime

import numpy as np
import scipy.io
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROMPT_TWO_BANDS = """
You are an expert in radar micro-motion target recognition. Please provide a professional textual description of the given HRRP image, focusing on two bright scattering lines (strong echo bands).

The output MUST follow this fixed structure:

- First sentence: describe that there are two bright scattering bands, and explicitly describe their relative spatial distance and vertical arrangement (e.g., close or far apart, upper/lower relationship).
- Second sentence: describe the intensity characteristics, stability, and structural appearance of each scattering band separately (upper band and lower band).
- Third sentence: describe the temporal evolution pattern of both bands, emphasizing their modulation behavior such as periodicity, synchronization, similarity, or differences.

Requirements:
- The output must be concise and professional.
- STRICTLY follow the 3-sentence structure above.
- Do NOT include any numerical values.
- Focus on spatial relationship, band-specific characteristics, and temporal variation trends.
- Avoid physical derivations; emphasize visual and pattern-level description only.
- The total length must be strictly between 50 and 60 words.
""".strip()


PROMPT_THREE_BANDS = """
You are an expert in radar micro-motion target recognition. Please provide a professional textual description of the given HRRP image, focusing on three bright scattering lines (strong echo bands).

The output MUST follow this fixed structure:

- First sentence: describe that there are three bright scattering bands, and explicitly describe their spatial arrangement and relative distances (e.g., upper, middle, lower; close or well-separated relationships).
- Second sentence: describe the intensity characteristics, stability, and visual appearance of each scattering band separately (upper, middle, and lower bands).
- Third sentence: describe the temporal evolution pattern of all bands, focusing on modulation behavior such as periodicity, synchronization, continuity, and possible differences among bands.

Requirements:
- The output must be concise and professional.
- STRICTLY follow the 3-sentence structure above.
- Do NOT include any numerical values.
- Emphasize spatial relationships, band-wise characteristics, and temporal variation trends.
- Avoid physical derivations; focus only on visual and pattern-level description.
- The total length must be strictly between 50 and 60 words.
""".strip()


def log_print(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def get_client(model_name, device_map="auto"):
    if model_name == "qwen3-vl-8b-local":
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
    elif model_name == "qwen3-vl-32b-local":
        model_path = "Qwen/Qwen3-VL-32B-Instruct"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return {"model": model, "processor": processor}


def load_hrrp_group(data_dir, freq, snr):
    train_path = os.path.join(data_dir, f"hrrp_train_data_{freq}_{snr}dB.mat")
    test_path = os.path.join(data_dir, f"hrrp_test_data_{freq}_{snr}dB.mat")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_mat = scipy.io.loadmat(train_path)
    test_mat = scipy.io.loadmat(test_path)
    train_data = np.abs(train_mat["train_data"])
    test_data = np.abs(test_mat["test_data"])

    data = np.concatenate((train_data, test_data), axis=2)
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min) * 255.0
    else:
        data = np.zeros_like(data, dtype=np.float32)

    split_names = ["train"] * train_data.shape[2] + ["test"] * test_data.shape[2]
    split_indices = list(range(train_data.shape[2])) + list(range(test_data.shape[2]))
    split_sizes = {"train": train_data.shape[2], "test": test_data.shape[2]}

    return data, split_names, split_indices, split_sizes


def get_prompt_for_sample(split_name, split_index, split_sizes):
    split_size = split_sizes[split_name]
    class_id = min((split_index * 4) // split_size, 3)
    if class_id in (0, 2):
        return "two_bands", PROMPT_TWO_BANDS
    return "three_bands", PROMPT_THREE_BANDS


def hrrp_image_to_data_url(hrrp_image):
    image_uint8 = np.clip(hrrp_image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_uint8).convert("L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def build_message(hrrp_image, prompt):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": hrrp_image_to_data_url(hrrp_image)},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


def convert_message_schema(messages):
    converted = []
    for msg in messages:
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                new_item = dict(item)
                if new_item.get("type") == "image_url":
                    image_url = new_item.pop("image_url", {})
                    new_item["type"] = "image"
                    new_item["image"] = image_url.get("url")
                new_content.append(new_item)
            new_msg["content"] = new_content
        elif isinstance(content, str):
            new_msg["content"] = [{"type": "text", "text": content}]
        converted.append(new_msg)
    return converted


def get_local_qwen_response(client, message, max_new_tokens=1024):
    model = client["model"]
    processor = client["processor"]

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    try:
        messages = convert_message_schema(message)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        description = output_text[0] if output_text else ""
        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = (
            int(generated_ids_trimmed[0].shape[0]) if generated_ids_trimmed else 0
        )
        total_tokens = prompt_tokens + completion_tokens
        status = "success"
    except Exception as exc:
        description = ""
        status = f"error: {repr(exc)}"

    return {
        "description": description,
        "status": status,
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def safe_model_name(model_name):
    return model_name.replace("/", "-").replace(":", "-")


def make_output_paths(output_dir, freq, snr, model_name):
    os.makedirs(output_dir, exist_ok=True)
    stem = f"hrrp_description_{freq}_{snr}dB_{safe_model_name(model_name)}"
    return os.path.join(output_dir, f"{stem}.jsonl"), os.path.join(output_dir, f"{stem}.npy")


def load_existing_ordered_prefix(jsonl_path, total_count):
    descriptions = [None] * total_count
    next_sample_id = 0

    if not os.path.exists(jsonl_path):
        return descriptions, next_sample_id

    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            if sample_id != next_sample_id:
                raise ValueError(
                    f"Existing JSONL is not an ordered prefix at line {line_number}: "
                    f"expected id={next_sample_id}, got id={sample_id}. "
                    "Please fix/remove the JSONL file or rerun without --resume."
                )
            if sample_id >= total_count:
                raise ValueError(
                    f"Existing JSONL has more rows than the current dataset: id={sample_id}, "
                    f"total_count={total_count}."
                )
            if record.get("status") != "success":
                raise ValueError(
                    f"Existing JSONL contains a non-success row at line {line_number}, "
                    f"id={sample_id}, status={record.get('status')!r}. "
                    "Rerun without --resume to overwrite failed output."
                )
            descriptions[sample_id] = record.get("description", "")
            next_sample_id += 1

    return descriptions, next_sample_id


def process_group(
    freq,
    snr,
    gpu_id,
    data_dir,
    output_dir,
    model_name,
    max_new_tokens,
    resume,
):
    device_map = f"cuda:{gpu_id}"
    jsonl_path, npy_path = make_output_paths(output_dir, freq, snr, model_name)
    start_time = time.time()

    try:
        log_print(f"group start | freq={freq} snr={snr} gpu={gpu_id}")
        data, split_names, split_indices, split_sizes = load_hrrp_group(data_dir, freq, snr)

        if resume:
            descriptions, start_sample_id = load_existing_ordered_prefix(
                jsonl_path,
                data.shape[2],
            )
            jsonl_mode = "a"
        else:
            descriptions = [None] * data.shape[2]
            start_sample_id = 0
            jsonl_mode = "w"

        client = get_client(model_name, device_map=device_map)
        log_print(
            f"model loaded | freq={freq} snr={snr} gpu={gpu_id} samples={data.shape[2]}"
        )

        with open(jsonl_path, jsonl_mode, encoding="utf-8") as fout:
            for sample_id in range(start_sample_id, data.shape[2]):
                sample_start = time.time()
                split_name = split_names[sample_id]
                split_index = split_indices[sample_id]
                prompt_type, prompt = get_prompt_for_sample(
                    split_name,
                    split_index,
                    split_sizes,
                )
                message = build_message(data[:, :, sample_id], prompt)

                response = get_local_qwen_response(
                    client=client,
                    message=message,
                    max_new_tokens=max_new_tokens,
                )
                description = response.get("description", "")
                status = response.get("status", "")
                descriptions[sample_id] = description

                record = {
                    "id": sample_id,
                    "freq": freq,
                    "snr": snr,
                    "split": split_name,
                    "split_index": split_index,
                    "prompt_type": prompt_type,
                    "description": description,
                    "status": status,
                    "model_name": model_name,
                    "tokens": response.get("tokens", {}),
                    "latency": time.time() - sample_start,
                }
                json.dump(record, fout, ensure_ascii=False)
                fout.write("\n")
                fout.flush()

                log_print(
                    f"sample done | freq={freq} snr={snr} id={sample_id} "
                    f"gpu={gpu_id} status={status}"
                )

        np.save(npy_path, np.array(descriptions, dtype=object))
        log_print(
            f"group finished | freq={freq} snr={snr} gpu={gpu_id} "
            f"elapsed={time.time() - start_time:.2f}s jsonl={jsonl_path} npy={npy_path}"
        )
        return 0
    except Exception as exc:
        log_print(f"group failed | freq={freq} snr={snr} gpu={gpu_id} error={repr(exc)}")
        log_print(traceback.format_exc())
        return 1


def parse_csv(value):
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_freq_snr_pairs(data_dir):
    pattern = re.compile(r"^hrrp_train_data_(?P<freq>.+)_(?P<snr>-?\d+(?:\.\d+)?)dB\.mat$")
    pairs = []

    if not os.path.isdir(data_dir):
        return pairs

    for filename in sorted(os.listdir(data_dir)):
        match = pattern.match(filename)
        if not match:
            continue
        freq = match.group("freq")
        snr = match.group("snr")
        test_path = os.path.join(data_dir, f"hrrp_test_data_{freq}_{snr}dB.mat")
        if os.path.exists(test_path):
            pairs.append((freq, snr))

    return pairs


def build_requested_pairs(data_dir, freqs, snrs):
    if freqs is None and snrs is None:
        return discover_freq_snr_pairs(data_dir)
    if not freqs or not snrs:
        raise ValueError("--freqs and --snrs must be provided together")
    return [(freq, snr) for freq in freqs for snr in snrs]


def group_process_entry(result_queue, task):
    freq, snr = task[0], task[1]
    exit_code = process_group(*task)
    result_queue.put(
        {
            "freq": freq,
            "snr": snr,
            "exit_code": exit_code,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HRRP text descriptions with local Qwen3-VL."
    )
    parser.add_argument("--data_dir", type=str, default="./hrrp_data")
    parser.add_argument("--output_dir", type=str, default="./hrrp_text_description")
    parser.add_argument("--freqs", type=str, default=None, help="Comma-separated values, e.g. L,S,X")
    parser.add_argument("--snrs", type=str, default=None, help="Comma-separated values, e.g. 0,5,10,20")
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3-vl-8b-local",
        choices=["qwen3-vl-8b-local", "qwen3-vl-32b-local"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing JSONL and skip samples with success status.",
    )
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    freqs = parse_csv(args.freqs)
    snrs = parse_csv(args.snrs)
    pairs = build_requested_pairs(args.data_dir, freqs, snrs)

    if not pairs:
        raise RuntimeError(
            f"No complete train/test HRRP pairs found in {args.data_dir}. "
            "Use --freqs and --snrs after placing files there."
        )

    gpu_slots = [
        gpu_id
        for gpu_id in range(args.num_gpus)
        for _ in range(args.workers_per_gpu)
    ]
    max_workers = len(gpu_slots)
    pending = list(pairs)
    available_gpus = list(gpu_slots)
    active = {}
    result_queue = mp.Queue()
    exit_codes = []

    log_print(f"found {len(pending)} group(s): {pending}")
    log_print(f"max concurrent processes: {max_workers}, gpu slots: {gpu_slots}")

    while pending or active:
        while pending and available_gpus:
            freq, snr = pending.pop(0)
            gpu_id = available_gpus.pop(0)
            task = (
                freq,
                snr,
                gpu_id,
                args.data_dir,
                args.output_dir,
                args.model_name,
                args.max_new_tokens,
                args.resume,
            )
            process = mp.Process(
                target=group_process_entry,
                args=(result_queue, task),
            )
            process.start()
            active[(freq, snr)] = (process, gpu_id)
            log_print(f"launched group process | freq={freq} snr={snr} gpu={gpu_id}")

        result = result_queue.get()
        key = (result["freq"], result["snr"])
        process, gpu_id = active.pop(key)
        process.join()
        available_gpus.append(gpu_id)
        exit_codes.append(result["exit_code"])

        log_print(
            f"group process joined | freq={key[0]} snr={key[1]} "
            f"gpu={gpu_id} exit_code={result['exit_code']}"
        )

    failed = sum(1 for code in exit_codes if code != 0)
    if failed:
        raise RuntimeError(f"{failed} group(s) failed")

    log_print("all groups finished")


if __name__ == "__main__":
    main()
