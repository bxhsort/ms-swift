import json
import os
from pathlib import Path
import re
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import json
import os
from tqdm import tqdm


def read_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(file_path: str, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_rubric(text: str):
    match = re.search(r"<Rubric>\s*(.*?)\s*</Rubric>", text, re.DOTALL)

    if not match:
        raise ValueError("没有找到 <Rubric>...</Rubric> 内容")

    rubric_text = match.group(1).strip()

    # 把 rubric 内容解析成 Python list/dict
    return json.loads(rubric_text)
def extract_eval(text: str):
    match = re.search(r"<Evaluation>\s*(.*?)\s*</Evaluation>", text, re.DOTALL)

    if not match:
        raise ValueError("没有找到 <Evaluation>...</Evaluation> 内容")

    rubric_text = match.group(1).strip()

    # 把 rubric 内容解析成 Python list/dict
    return json.loads(rubric_text)

SYSTEM_PROMPT = """
You are a professional image editing evaluator. To evaluate editing results comprehensively and carefully, follow the two-stage evaluation strategy below:

Stage 1: Generate evaluation points covering three dimensions: Instruction Following, Visual Consistency, and Visual Quality. Each evaluation point must be a clear, specific, verifiable QA-style question, written concisely and directly.
Stage 2: Evaluate each generated evaluation point one by one and assign a score of 0 or 1 to each point.

You will receive:
    Input Images: [Path1]
    Instruction: [Input Instruction]
    Edited Image: [Path2]

Stage 1: Generate Evaluation Points

The evaluation points must be divided into three dimensions:

1. Instruction Following: Compare the original image with the edited image and determine whether the editing result accurately follows the given instruction. For editing instructions that require implicit reasoning, also use relevant common sense, world knowledge, and applicable rules to evaluate whether the result satisfies both explicit requirements and implicit constraints.

2. Visual Consistency: Compare the original image with the edited image and determine whether any unintended changes occurred beyond the given instruction.

3. Visual Quality: Judge whether the edited image is clear, natural, free of obvious artifacts, and free of structural issues.

Rules for generating evaluation points:

* Generate 8 to 14 evaluation points in total. Typically include 1 to 4 points for Instruction Following, up to 5 for complex instructions; 5 to 6 points for Visual Consistency; and 2 to 4 points for Visual Quality.
* Instruction Following should focus only on whether the intent of the edit instruction is achieved. Do not evaluate visual consistency or visual quality in this dimension.
* For Instruction Following, if the edit instruction contains multiple sub-instructions, such as “add an apple and remove the bag,” each sub-instruction must be treated as an independent evaluation point. Also analyze the potential intent and relevant world knowledge behind each sub-instruction, and construct evaluation points accordingly.
* Visual Consistency should evaluate only whether content outside the edit instruction remains consistent, including unedited objects, background areas, overall lighting, perspective, composition, and spatial layout. If only a single attribute of an object is edited, the object’s other attributes should remain consistent and may be used as Visual Consistency evaluation points. Do not classify the naturalness, deformation, floating appearance, or poor blending of newly added or replaced objects as Visual Consistency issues unless the instruction explicitly requires those attributes; otherwise, classify them under Visual Quality.
* Visual Consistency evaluation points should prioritize three types of content:
  1. Unchanged attributes of the edited object, such as shape, size, orientation, pose, or structure;
  2. One to three semantically related but unedited objects, where each evaluation point should include a clear spatial position, visual attribute, and distinguishing feature;
  3. Two to three background objects unrelated to the edit instruction and not overlapping with the previous category, with descriptions of their position and appearance.
     Finally, add one to two global consistency questions if needed, checking whether lighting, perspective, composition, or spatial layout changed unexpectedly.
* Visual Quality should evaluate only the image formation quality and visual naturalness of the edited image. It should not judge whether the instruction was completed, nor whether unedited content remains consistent. Evaluation points should focus on four types of issues:
  1. Whether the edited image has unexpected blur, noise, loss of sharpness, or other image quality degradation compared with the original;
  2. Whether the edited, changed, or newly added object naturally integrates into the scene and is free of obvious artifacts, abnormal edges, deformation, or poor blending;
  3. Whether there are structural problems, such as broken fingers, extra hands, deformed limbs or facial features, or damaged object structures;
  4. If the instruction involves text modification, whether the text is blurry, garbled, misspelled, or unreadable.
* Avoid repeatedly evaluating the same object or the same attribute within the same dimension unless the local regions or attributes are clearly distinct.

After generating comprehensive evaluation points based on the above rules, proceed to Stage 2 and evaluate each point one by one.

Stage 2: Evaluation

Evaluation rules:

* For each evaluation point, such as “Has the scene changed from indoors to outdoors?”, assign only 0 for not met or 1 for met. For edits involving ranges, such as far to near, left to right, fat to thin, or small to large, a significant change is required to count as met unless the instruction specifies the magnitude.
* If you are uncertain about an evaluation point, assign 0 and reduce its weight when calculating the final score.
* The final score should not rely solely on the average of the evaluation points. The final score may be any value from 0 to 10, including decimals.
* If an evaluation point contradicts the edit instruction or has a causal relationship with it—for example, if a point evaluates global lighting consistency but the object removed according to the instruction affects that consistency—then ignore that point or partially score it based only on unaffected regions when calculating the score.
* Pay special attention to newly revealed areas caused by object movement or changes in occlusion, while also focusing on the consistency of originally unoccluded parts.
* If the edited image is nearly identical to the original image or the edited object has not changed, meaning no edit was performed, assign a score of 0.
* For consistency evaluation, all Visual Consistency evaluation points should receive equal weight.
* For positional changes, such as left to right, output the bounding box coordinates of the edited object in both the original and edited images during reasoning. A significant shift is required; minor movement is not valid.
* When evaluating human pose, strictly determine left and right based on the person’s own orientation.
* Also consider unmentioned elements, such as the main subject, other objects, and global lighting. If they should not change but are inconsistent, apply heavier penalties.
* Check visual quality issues: the image should not have white borders, over-sharpening, over-saturation, or color cast. White borders should receive a minor deduction. Artifacts or structural issues, such as broken limbs, should receive heavy penalties.
* For Visual Quality, if both the original and edited images have the quality issue described by an evaluation point, reduce the weight of that point in the final score.
* For tasks involving world-knowledge reasoning, appropriately increase the weight of Instruction Following in the final score.

Strictly follow these steps in order:

1. Output the generated evaluation points in the format [{'question': ..., 'dimension': ...}], and place them between <Rubric> and </Rubric> tokens.
2. For each evaluation point, evaluate and score it based on the comparison between the original image, the edited image, and the edit instruction. Strictly follow the rules above. Calculate the average score across all evaluation points, round it to two decimal places, and output it as {"average_score": <score>}.
3. Based on the above, assign a final score to the edited image from 0 to 10:
   * 0 = completely unusable, with serious issues in the edit;
   * 5 = partially usable, with some minor attempt, but the editing quality is far from sufficient;
   * 8 = nearly usable, with only minor defects, slight visual inconsistencies, or minor instruction deviations.
4. If positional changes are involved, output bounding box coordinates during reasoning and determine whether the edit is valid.
5. Confirm whether the edit was actually performed by comparing the differences between the original image and the edited image.
6. Explain how average_score is calculated. For example: “Instruction Following: 1+0+1=2; Visual Consistency: 1+1=2; Visual Quality: 0+1=1; total: 5/7, so average_score=7.14.” Then, based on the score of each evaluation point, separately explain the rationale for final_score across different dimensions, such as Instruction Following, Visual Consistency, Visual Quality, and any other relevant dimensions. Finally, summarize the overall justification for final_score.

Output

Strictly output the result in the following format:

<Rubric>
[
  {
    "id": "q1",
    "dimension": "Instruction Following",
    "question": "Is the background changed into a recognizable indoor Potala Palace setting in Tibet?"
  },
  {
    "id": "q2",
    "dimension": "Visual Consistency",
    "question": "Are the couple, clothing, pose, composition, and original black-and-white style preserved without unintended changes?"
  },
  {
    "id": "q3",
    "dimension": "Visual Quality",
    "question": "Is the edited image visually natural, sharp, well-blended, and free from obvious artifacts or structural distortions?"
  }
]
</Rubric>

<Evaluation>
{
  "eval_points": [
    {
      "id": "q1",
      "score": 0,
      "evidence": "The background includes some Tibetan-style red decoration but is not recognizable as an indoor Potala Palace setting."
    },
    {
      "id": "q2",
      "score": 0,
      "evidence": "The couple's pose and composition are mostly preserved, but the image was unintentionally colorized and the clothing and makeup changed noticeably."
    },
    {
      "id": "q3",
      "score": 1,
      "evidence": "The image is generally sharp, cleanly blended, and free from obvious structural deformation."
    }
  ],
  "average_score": 3.33,
  "final_score": 3,
  "reason": ""
}
</Evaluation>
"""

IMAGE_DIR="/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_Bench/EditReward-Bench/Images"
MODEL = "/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/ms-swift/output/qwen3vl_8b_test_it2i/v1-20260621-032217/checkpoint-100-merged"
DATA_path = "/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_Bench/EditReward-Bench/metadata.json"
INDEX = 0

data = read_json(DATA_path)

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, dtype="auto", device_map="auto"
)

 

processor = AutoProcessor.from_pretrained(MODEL)



def evaluate_output(item, output_index):
    input_image = os.path.join(IMAGE_DIR, item["source_image"])
    output_image = os.path.join(IMAGE_DIR, item["output_images"][output_index])

    prompt = (
        SYSTEM_PROMPT
        .replace("[Path1]", "<image>\n")
        .replace("[Input Instruction]", item["instruction"])
        .replace("[Path2]", "<image>\n")
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": input_image},
            {"type": "image", "image": output_image},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]

    evals = extract_eval(output_text)
    rubric = extract_rubric(output_text)

    id_to_dimension = {
        point["id"]: point["dimension"]
        for point in rubric
    }

    dimension_scores = {}
    for point in evals["eval_points"]:
        dimension = id_to_dimension[point["id"]]
        dimension_scores.setdefault(dimension, []).append(point["score"])

    target_dimension = item["dimension"]
    if target_dimension == "overall":
        return evals["average_score"], output_text

    scores = dimension_scores[target_dimension]
    return sum(scores) / len(scores), output_text
  
  
pbar = tqdm(
    enumerate(data),
    total=len(data),
    desc="Evaluating",
    colour="green",
)

results = []

correct = 0
ties = 0
valid_num = 0

for index, item in pbar:
    if len(item["output_images"]) < 2:
        print(f"跳过第 {index} 条：output_images 少于两个")
        continue

    try:
        score_0, response_0 = evaluate_output(item, 0)
        score_1, response_1 = evaluate_output(item, 1)

        is_correct = score_0 > score_1
         

        correct += int(is_correct)
        
        valid_num += 1

        strict_acc = correct / valid_num

        pbar.set_postfix({
            "acc": f"{strict_acc:.4f}",
            "score0": f"{score_0:.2f}",
            "score1": f"{score_1:.2f}",
        })

        results.append({
            "index": index,
            "dimension": item["dimension"],
            "instruction": item["instruction"],
            "output_image_0": item["output_images"][0],
            "output_image_1": item["output_images"][1],
            "score_0": score_0,
            "score_1": score_1,
            "correct": is_correct,
            "response_0": response_0,
            "response_1": response_1,
        })
        write_json("/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_Bench/EditReward-Bench/Results/pairwise_eval_results.json",results)

    except Exception as error:
        tqdm.write(f"第 {index} 条处理失败：{error}")
write_json( "/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_Bench/EditReward-Bench/Results/pairwise_eval_results.json",results)