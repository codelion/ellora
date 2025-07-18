# 🎯 Ellora: Enhancing LLMs with LoRA

[![GitHub](https://img.shields.io/github/license/codelion/ellora)](https://github.com/codelion/ellora/blob/main/LICENSE)
[![Models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/models?other=ellora)

> **Ellora** (Enhancing LLMs with LoRA) is a collection of standardized, high-quality LoRA recipes for enhancing Large Language Model capabilities. Instead of building new frameworks, we focus on creating reproducible training methodologies that work with existing infrastructure.

## 🌟 Philosophy

The LLM ecosystem has amazing infrastructure (LoRAX, PEFT, vLLM), but lacks **standardized, high-quality capability adapters**. Ellora bridges this gap by providing:

- 📋 **Recipes, not frameworks** - Reproducible training methodologies
- 🎯 **Quality-first approach** - Rigorous evaluation and benchmarking  
- 🔄 **Self-supervised data generation** - No dependency on external datasets
- 🏗️ **Infrastructure agnostic** - Works with existing tools (PEFT, LoRAX, etc.)
- 🌍 **Community-driven** - Open recipes for the ecosystem

## 🍳 Available Recipes

### Recipe #1: Accuracy Recovery LoRA
**Problem**: Quantized models (INT4/INT8) lose accuracy compared to FP16 versions  
**Solution**: Self-distillation LoRA adapter using Magpie-generated data

- 🎯 **Goal**: <5% performance degradation from FP16 baseline
- 💾 **Memory**: ~75% reduction in model size  
- ⚡ **Speed**: 2-3x faster inference than FP16
- 📊 **Method**: Teacher (FP16) → Student (INT4+LoRA) distillation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codelion/ellora/blob/main/Ellora_Recipe_1_Self_Distillation_For_Quantization_Recovery.ipynb)

**Key Innovation**: Uses [Magpie](https://arxiv.org/abs/2406.08464) self-data generation for perfect domain alignment - no external datasets needed!

#### Quick Start
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
)

# Load accuracy recovery adapter
model = PeftModel.from_pretrained(model, "codelion/Qwen3-0.6B-accuracy-recovery-lora")

# Use normally - now with recovered accuracy!
```

#### Results
| Model | Perplexity | Memory | Speed | Status |
|-------|------------|--------|-------|---------|
| FP16 Baseline | 1.97 | 1.0GB | 1.0x | ✅ |
| INT4 Raw | 2.40 (+21.8%) | 0.25GB | 3.2x | ⚠️ |
| INT4 + Ellora | 2.09 (+5.7%) | 0.28GB | 3.0x | ✅ |

### Recipe #2: Reasoning LoRA with GRPO
**Problem**: LLMs often lack structured thinking patterns for complex reasoning  
**Solution**: GRPO-trained adapter that teaches chain-of-thought with `<think></think>` tags

- 🧠 **Goal**: Enhance reasoning capabilities through preference learning
- 📝 **Method**: GRPO (Group Relative Policy Optimization) with self-rewarding
- 🎯 **Feature**: Teaches structured thinking with clear reasoning steps
- 💡 **Output**: Models that show their reasoning process transparently

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codelion/ellora/blob/main/Ellora_Recipe_2_Reasoning_LoRA_with_Self-Rewarding_GRPO.ipynb)

**Key Innovation**: Self-generated preference data with automated quality scoring - no need for human annotations or external preference datasets!

#### Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Load reasoning adapter
model = PeftModel.from_pretrained(model, "codelion/gemma-3-1b-it-reasoning-grpo-lora")

# Use with thinking prompt
prompt = '''Think step by step and use <think></think> tags to show your reasoning process.

Problem: If a train travels 120 miles in 2 hours, then increases its speed by 30 mph for the next hour, how many total miles does it travel?

Response:'''

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Results
| Model | Thinking Usage | Quality Score | Training Method | Status |
|-------|----------------|---------------|-----------------|---------|
| Gemma-3-1B Base | 0% | 3.2 | - | ⚠️ |
| Gemma-3-1B + Ellora | 60% | 5.6 | GRPO | ✅ |

## 🏆 Model Zoo

All models trained using Ellora recipes are available on HuggingFace:

[![Models](https://img.shields.io/badge/🤗_Explore_Models-yellow?style=for-the-badge)](https://huggingface.co/models?other=ellora)

### Featured Models
- [`codelion/Qwen3-0.6B-accuracy-recovery-lora`](https://huggingface.co/codelion/Qwen3-0.6B-accuracy-recovery-lora) - Accuracy recovery for Qwen3-0.6B
- [`codelion/gemma-3-1b-it-reasoning-grpo-lora`](https://huggingface.co/codelion/gemma-3-1b-it-reasoning-grpo-lora) - Reasoning enhancement for Gemma-3-1B
- More models coming as we test recipes across different model families!

## 🔬 Research & Citations

If you use Ellora recipes in your research, please cite:

```bibtex
@misc{ellora2024,
  title={Ellora: Enhancing LLMs with LoRA - Standardized Recipes for Capability Enhancement},
  author={Asankhaya Sharma},
  year={2024},
  url={https://github.com/codelion/ellora}
}
```

### Key Papers & Inspirations
- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Magpie**: [Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
- **QLoRA**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
