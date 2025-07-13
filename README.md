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

## 🏆 Model Zoo

All models trained using Ellora recipes are available on HuggingFace:

[![Models](https://img.shields.io/badge/🤗_Explore_Models-yellow?style=for-the-badge)](https://huggingface.co/models?other=ellora)

### Featured Models
- [`codelion/Qwen3-0.6B-accuracy-recovery-lora`](https://huggingface.co/codelion/Qwen3-0.6B-accuracy-recovery-lora) - Accuracy recovery for Qwen3-0.6B
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
