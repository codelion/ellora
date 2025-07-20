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

## 📚 Recipe Collection

| Recipe | Purpose | Key Achievement | Jump to |
|--------|---------|-----------------|----------|
| **#1: Accuracy Recovery** | Restore quantized model performance | <5% degradation from FP16 | [Details](#recipe-1-accuracy-recovery-lora) |
| **#2: Reasoning Enhancement** | Add structured thinking with `<think>` tags | 60% thinking usage, 75% quality boost | [Details](#recipe-2-reasoning-lora-with-grpo) |
| **#3: Tool Calling** | Enable effective development tool usage | 80% success rate on complex tasks | [Details](#recipe-3-tool-calling-lora) |
| **#4: Context Extension** | Expand from 32K to 2M tokens | 61x context increase for full repos | [Details](#recipe-4-progressive-context-extension-lora) |

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

### Recipe #3: Tool Calling LoRA
**Problem**: LLMs struggle with effective tool usage for code exploration  
**Solution**: Hybrid training with Magpie scenarios + real tool execution results

- 🛠️ **Goal**: Teach models to use development tools effectively
- 🔄 **Method**: Generate scenarios with Magpie, execute on real codebases
- 🎯 **Feature**: OpenAI-compatible function calling format
- 💻 **Tools**: File operations, search, code navigation, and more

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codelion/ellora/blob/main/Ellora_Recipe_3_Enhanced_Tool_Calling_and_Code_Understanding.ipynb)

**Key Innovation**: Combines synthetic scenario diversity with real execution feedback - ensuring models learn authentic tool usage patterns!

### Recipe #4: Progressive Context Extension LoRA
**Problem**: Base models limited to 32K context, need 2M tokens for large repositories  
**Solution**: Progressive curriculum learning with vLLM + Unsloth hybrid approach

- 📈 **Goal**: Extend context from 32K to 2M tokens (61x increase)
- 🎓 **Method**: Curriculum learning across 4 stages (32K → 128K → 512K → 2M)
- ⚡ **Innovation**: vLLM for fast data generation, Unsloth for memory-efficient training
- 🔍 **Feature**: Single LoRA adapter progressively learns longer contexts

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codelion/ellora/blob/main/Ellora_Recipe_4_Repository_Context_LoRA.ipynb)

**Key Innovation**: Hybrid optimization combining vLLM's inference speed with Unsloth's training efficiency - achieving 61x context extension with minimal compute!

#### Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Load progressive context adapter
model = PeftModel.from_pretrained(model, "codelion/qwen2-5-coder-0-5b-instruct-progressive-2000k-lora")

# Use with 2M token context - perfect for large repositories!
long_context_prompt = "Analyze this entire repository..." # Up to 2M tokens
inputs = tokenizer(long_context_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024)
```

#### Results
| Model | Context Limit | Max Files | Use Case | Status |
|-------|---------------|-----------|----------|---------|
| Qwen2.5-Coder Base | 32K tokens | ~10-20 files | Small projects | ⚠️ |
| + Stage 0 LoRA | 32K tokens | ~10-20 files | Single module analysis | ✅ |
| + Stage 1 LoRA | 128K tokens | ~50-100 files | Medium repositories | ✅ |
| + Stage 2 LoRA | 512K tokens | ~200-500 files | Large codebases | ✅ |
| + Stage 3 LoRA | 2M tokens | ~1000+ files | Entire repositories | ✅ |

## 🏆 Model Zoo

All models trained using Ellora recipes are available on HuggingFace:

[![Models](https://img.shields.io/badge/🤗_Explore_Models-yellow?style=for-the-badge)](https://huggingface.co/models?other=ellora)

### Featured Models
- [`codelion/Qwen3-0.6B-accuracy-recovery-lora`](https://huggingface.co/codelion/Qwen3-0.6B-accuracy-recovery-lora) - Accuracy recovery for Qwen3-0.6B
- [`codelion/gemma-3-1b-it-reasoning-grpo-lora`](https://huggingface.co/codelion/gemma-3-1b-it-reasoning-grpo-lora) - Reasoning enhancement for Gemma-3-1B
- [`codelion/Llama-3.2-1B-Instruct-tool-calling-lora`](https://huggingface.co/codelion/Llama-3.2-1B-Instruct-tool-calling-lora) - Tool calling for Llama-3.2-1B
- [`codelion/qwen2-5-coder-0-5b-instruct-progressive-2000k-lora`](https://huggingface.co/codelion/qwen2-5-coder-0-5b-instruct-progressive-2000k-lora) - 2M context extension for Qwen2.5-Coder-0.5B
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
