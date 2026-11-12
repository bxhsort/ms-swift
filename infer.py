import argparse
import json


DEFAULT_MODEL_PATH = (
    "/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/ms-swift/output/qwen3_5_9b_score/v2-20260720-145452/checkpoint-156-merged"
)


def load_jsonl_row(path, index):
    with open(path, "r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"index {index} is out of range for {path}")


def split_prompt_and_target(row):
    messages = row.get("messages") or []
    prompt_messages = []
    target = None
    for message in messages:
        if message.get("role") == "assistant":
            target = message.get("content", "")
            break
        prompt_messages.append(message)
    if not prompt_messages:
        raise ValueError("missing prompt messages")
    return prompt_messages, target


def clean_text_messages(messages):
    cleaned = []
    for message in messages:
        content = str(message.get("content", "")).replace("<image>", "").strip()
        cleaned.append({"role": message.get("role", "user"), "content": content})
    return cleaned


def apply_chat_template_compat(template_obj, messages, args):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": args.enable_thinking,
    }
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget

    try:
        return template_obj.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("thinking_budget", None)
        try:
            return template_obj.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return template_obj.apply_chat_template(messages, **kwargs)


def strip_thinking_text(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        return text.split("<think>", 1)[0].strip()
    return text


def run_text_model(args, messages):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    text = apply_chat_template_compat(tokenizer, clean_text_messages(messages), args)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    return strip_thinking_text(response) if args.strip_thinking else response


def load_vl_classes():
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText

        return AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        return AutoProcessor, AutoModelForVision2Seq


def build_vl_messages(messages, images):
    vl_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if role != "user":
            vl_messages.append({"role": role, "content": content.replace("<image>", "").strip()})
            continue

        text = content.replace("<image>", "").strip()
        parts = [{"type": "image", "image": image} for image in images]
        parts.append({"type": "text", "text": text})
        vl_messages.append({"role": role, "content": parts})
    return vl_messages


def run_vl_model(args, messages, image_paths):
    import torch
    from PIL import Image

    AutoProcessor, AutoModel = load_vl_classes()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    vl_messages = build_vl_messages(messages, image_paths)
    text = apply_chat_template_compat(processor, vl_messages, args)
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )

    generated_ids = outputs[:, inputs["input_ids"].shape[-1] :]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return strip_thinking_text(response) if args.strip_thinking else response


def parse_args():
    parser = argparse.ArgumentParser(description="Run one ms-swift JSONL sample through a merged checkpoint.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset_jsonl", default="/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_Data/ESData/checklist_refine_ans_msswift.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["text", "vl"],
        default="vl",
        help="Use text for CausalLM checkpoints; use vl for image-text checkpoints.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen-style thinking mode in apply_chat_template. Default is disabled.",
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=None,
        help="Optional thinking token budget if the tokenizer/processor supports it.",
    )
    parser.add_argument(
        "--strip_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip generated <think>...</think> text before printing the response.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the selected sample and target answer without loading the model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    row = load_jsonl_row(args.dataset_jsonl, args.index)
    prompt_messages, target = split_prompt_and_target(row)
    image_paths = row.get("images") or []

    print(f"sample index: {args.index}")
    print(f"image count: {len(image_paths)}")
    for image in image_paths:
        print(f"image: {image}")
    print("\n===== Ground Truth =====")
    print(target or "")

    if args.dry_run:
        return

    if args.mode == "text":
        response = run_text_model(args, prompt_messages)
    else:
        response = run_vl_model(args, prompt_messages, image_paths)

    print("\n===== Model Response =====")
    print(response)


if __name__ == "__main__":
    main()
