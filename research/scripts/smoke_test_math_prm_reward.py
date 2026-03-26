#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from examples.reward_function.math_prm_step_reward import compute_score


DEFAULT_PROBLEM = (
    "Sue lives in a fun neighborhood. One weekend, the neighbors decided to play a prank on Sue. "
    "On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard. "
    "On Saturday morning, the neighbors took back one third of the flamingos, painted them white, "
    "and put these newly painted white flamingos back out on Sue's front yard. Then, on Sunday morning, "
    "they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more "
    "pink plastic flamingos were out than white plastic flamingos?"
)

DEFAULT_RESPONSE = r"""<think>
Start with 18 pink flamingos on Friday.

On Saturday, one third of 18 is 6, so 6 flamingos are repainted white and 12 pink remain.

That means after Saturday there are 12 pink flamingos and 6 white flamingos in the yard.

On Sunday, 18 more pink flamingos are added, so the total becomes 30 pink flamingos and 6 white flamingos.

The difference is 30 - 6 = 24, so the final answer is \boxed{24}.
</think>\boxed{24}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the math PRM reward function on one example.")
    parser.add_argument("--problem", type=str, default=DEFAULT_PROBLEM, help="Problem text.")
    parser.add_argument("--response", type=str, default=DEFAULT_RESPONSE, help="Model response text.")
    parser.add_argument("--ground-truth", type=str, default="24", help="Ground-truth boxed answer content.")
    parser.add_argument(
        "--prm-model-path",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="HF model id or local path for the PRM.",
    )
    parser.add_argument("--prm-weight", type=float, default=0.1, help="Weight applied to PRM product score.")
    parser.add_argument("--format-weight", type=float, default=0.1, help="Weight applied to format score in raw math reward.")
    parser.add_argument("--max-prm-length", type=int, default=4096)
    parser.add_argument("--max-prm-steps", type=int, default=8)
    parser.add_argument("--min-step-chars", type=int, default=8)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto", help="Transformers device_map passed to PRM.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional JSON file with keys: source_prompt/problem, response, ground_truth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_json is not None:
        payload = json.loads(args.input_json.read_text())
        problem = payload.get("source_prompt") or payload.get("problem") or args.problem
        response = payload.get("response", args.response)
        ground_truth = payload.get("ground_truth", args.ground_truth)
    else:
        problem = args.problem
        response = args.response
        ground_truth = args.ground_truth

    reward_input = [
        {
            "source_prompt": problem,
            "prompt": problem,
            "response": response,
            "response_length": len(response),
            "ground_truth": ground_truth,
        }
    ]

    scores = compute_score(
        reward_input,
        prm_model_path=args.prm_model_path,
        prm_weight=args.prm_weight,
        format_weight=args.format_weight,
        max_prm_length=args.max_prm_length,
        max_prm_steps=args.max_prm_steps,
        min_step_chars=args.min_step_chars,
        trust_remote_code=True,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )

    print(json.dumps({"input": reward_input[0], "score": scores[0]}, indent=2))


if __name__ == "__main__":
    main()
