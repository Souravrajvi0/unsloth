"""GRPO training test for the `fast_inference=True` vLLM rollout path.

Requires a CUDA or XPU GPU with vLLM installed.

Usage:
    python tests/fast_inference/test_fast_inference.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from tests.utils import header_footer_context


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 512
MAX_COMPLETION_LENGTH = 256
LOAD_IN_4BIT = False
GPU_MEMORY_UTILIZATION = 0.5
LORA_RANK = 16
NUM_GENERATIONS = 2
MAX_STEPS = 3
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.8

SYSTEM_PROMPT = "Respond concisely."
QUESTIONS = [
    "What is the capital of France?",
    "Name a primary color.",
    "What is 2 + 2?",
    "Write the word 'hello'.",
]
PROMPTS = [
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    for q in QUESTIONS
]


def length_reward_func(completions, **kwargs) -> list[float]:
    """Simple reward: prefer non-empty completions."""
    responses = [completion[0]["content"] for completion in completions]
    return [1.0 if len(r) > 0 else 0.0 for r in responses]


def get_unsloth_peft_model(
    model,
    lora_rank: int,
    target_modules = "all-linear",
    use_gradient_checkpointing = False,
    random_state: int = 42,
):
    return FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = target_modules,
        lora_alpha = lora_rank,
        use_gradient_checkpointing = use_gradient_checkpointing,
        random_state = random_state,
    )


if __name__ == "__main__":
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    with header_footer_context("Load model (fast_inference=True)"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_NAME,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = torch.bfloat16,
            load_in_4bit = LOAD_IN_4BIT,
            fast_inference = True,
            gpu_memory_utilization = GPU_MEMORY_UTILIZATION,
        )

    assert hasattr(model, "vllm_engine"), "model.vllm_engine not set — fast_inference=True failed"
    print("vllm_engine attached")

    model = get_unsloth_peft_model(
        model,
        lora_rank = LORA_RANK,
        target_modules = target_modules,
        use_gradient_checkpointing = False,
        random_state = 42,
    )

    prompt = tokenizer.apply_chat_template(PROMPTS[0], tokenize = False, add_generation_prompt = True)
    with header_footer_context("Test Prompt"):
        print(prompt)

    dataset = Dataset.from_dict({"prompt": PROMPTS})

    with header_footer_context("GRPO config and trainer"):
        training_args = GRPOConfig(
            learning_rate = 5e-6,
            optim = "adamw_torch",
            temperature = TEMPERATURE,
            top_k = 50,
            logging_steps = 1,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = NUM_GENERATIONS,
            num_generations = NUM_GENERATIONS,
            max_completion_length = MAX_COMPLETION_LENGTH,
            max_steps = MAX_STEPS,
            report_to = "none",
        )
        print(training_args)

        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = [length_reward_func],
            args = training_args,
            train_dataset = dataset,
        )

    with header_footer_context("GRPO train (vLLM rollout)"):
        trainer_stats = trainer.train()
        print(trainer_stats)

    assert trainer_stats is not None, "trainer.train() is None"
