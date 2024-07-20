import os
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, HfArgumentParser, AutoImageProcessor
from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import logger_setting, log_trainable_params
from tinyllava.model import ModelArguments, DataArguments
from tinyllava.data.dataset import make_supervised_data_module

DATA_PATH = "/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json"
IMAGE_PATH = "/home/ai/data/llava/dataset"
MODEL_MAX_LENGTH = 3072
OUTPUT_DIR = "/mnt/data/sata/yinghu/checkpoints/llava_factory/custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora"
PRETRAINED_MODEL_PATH = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
DEEPSPEED_CONFIG = "./scripts/zero2.json"

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    # Additional settings from command line
    model_arguments.tune_type_llm = "lora"
    model_arguments.tune_type_vision_tower = "frozen"
    model_arguments.tune_type_connector = "full"
    model_arguments.lora_r = 128
    model_arguments.lora_alpha = 256
    model_arguments.mm_vision_select_layer = -2
    model_arguments.conv_version = "phi"
    model_arguments.tune_vision_tower_from_layer = 0
    data_arguments.image_aspect_ratio = "square"
    data_arguments.image_folder = IMAGE_PATH
    data_arguments.data_path = DATA_PATH
    data_arguments.is_multimodal = True
    data_arguments.lazy_preprocess = True

    training_arguments.output_dir = OUTPUT_DIR
    training_arguments.model_max_length = MODEL_MAX_LENGTH
    training_arguments.num_train_epochs = 1
    training_arguments.per_device_train_batch_size = 4
    training_arguments.per_device_eval_batch_size = 4
    training_arguments.gradient_accumulation_steps = 8
    training_arguments.evaluation_strategy = "no"
    training_arguments.save_strategy = "steps"
    training_arguments.save_steps = 50000
    training_arguments.save_total_limit = 1
    training_arguments.learning_rate = 1e-4
    training_arguments.weight_decay = 0.
    training_arguments.warmup_ratio = 0.03
    training_arguments.lr_scheduler_type = "cosine"
    training_arguments.logging_steps = 1
    training_arguments.tf32 = False
    training_arguments.gradient_checkpointing = True
    training_arguments.dataloader_num_workers = 8
    training_arguments.report_to = "wandb"  # Use wandb for logging
    training_arguments.run_name = "custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora"
    training_arguments.deepspeed = DEEPSPEED_CONFIG  # Add DeepSpeed configuration
    training_arguments.fp16 = True
    training_arguments.group_by_modality_length = False

    # Initialize wandb
    wandb.init(project="your-wandb-project-name", name=training_arguments.run_name)

    logger_setting(training_arguments.output_dir)
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    load_settings(model_arguments, data_arguments, training_arguments)

    # Load pretrained checkpoint
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL_PATH, trust_remote_code=True)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, use_fast=False, model_max_length=config.tokenizer_model_max_length, padding_side=config.tokenizer_padding_side)
    model.tokenizer = tokenizer
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    data_arguments.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name_or_path)
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_arguments)

    log_trainable_params(model)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=data_module["train"],
        eval_dataset=data_module["eval"],
        tokenizer=tokenizer,
        data_collator=data_module["data_collator"]
    )

    trainer.train()
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    os.environ["DATA_PATH"] = DATA_PATH
    os.environ["IMAGE_PATH"] = IMAGE_PATH
    os.environ["MODEL_MAX_LENGTH"] = str(MODEL_MAX_LENGTH)
    os.environ["OUTPUT_DIR"] = OUTPUT_DIR
    train()
