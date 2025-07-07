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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dwRbvxtSpCiKGgvbOwM9jboE5v8WgAd-?usp=sharing)

**Key Innovation**: Uses [Magpie](https://arxiv.org/abs/2406.08464) self-data generation for perfect domain alignment - no external datasets needed!

#### Quick Start
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
)

# Load accuracy recovery adapter
model = PeftModel.from_pretrained(model, "codelion/Qwen2-0.5B-Instruct-accuracy-recovery-lora")

# Use normally - now with recovered accuracy!
```

#### Results
| Model | Perplexity | Memory | Speed | Status |
|-------|------------|--------|-------|---------|
| FP16 Baseline | 2.45 | 1.0GB | 1.0x | ✅ |
| INT4 Raw | 2.89 (+18%) | 0.25GB | 3.2x | ⚠️ |
| INT4 + Ellora | 2.51 (+2.4%) | 0.28GB | 3.0x | ✅ |

### 🚧 Coming Soon

- **Recipe #2: Reasoning LoRA** - Enhanced logical reasoning using GRPO
- **Recipe #3: Tool Calling LoRA** - Function calling and API interaction
- **Recipe #4: Safety LoRA** - Improved safety and alignment  
- **Recipe #5: Multilingual LoRA** - Enhanced multilingual performance

## 🏆 Model Zoo

All models trained using Ellora recipes are available on HuggingFace:

[![Models](https://img.shields.io/badge/🤗_Explore_Models-yellow?style=for-the-badge)](https://huggingface.co/models?other=ellora)

### Featured Models
- [`codelion/Qwen2-0.5B-Instruct-accuracy-recovery-lora`](https://huggingface.co/codelion/Qwen2-0.5B-Instruct-accuracy-recovery-lora) - Accuracy recovery for Qwen2-0.5B
- More models coming as we test recipes across different model families!

## 🔬 Research & Citations

If you use Ellora recipes in your research, please cite:

```bibtex
@misc{ellora2024,
  title={Ellora: Enhancing LLMs with LoRA - Standardized Recipes for Capability Enhancement},
  author={Asankahya Sharma},
  year={2024},
  url={https://github.com/codelion/ellora}
}
```

### Key Papers & Inspirations
- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Magpie**: [Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
- **QLoRA**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
