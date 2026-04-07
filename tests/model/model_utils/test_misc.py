# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llamafactory.model.model_utils.misc import find_all_linear_modules, find_expanded_modules


HF_TOKEN = os.getenv("HF_TOKEN")


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_expanded_modules():
    config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    expanded_modules = find_expanded_modules(model, ["q_proj", "v_proj"], num_layer_trainable=4)
    assert expanded_modules == [
        "model.layers.7.self_attn.q_proj",
        "model.layers.7.self_attn.v_proj",
        "model.layers.15.self_attn.q_proj",
        "model.layers.15.self_attn.v_proj",
        "model.layers.23.self_attn.q_proj",
        "model.layers.23.self_attn.v_proj",
        "model.layers.31.self_attn.q_proj",
        "model.layers.31.self_attn.v_proj",
    ]


def test_find_all_linear_modules_includes_conv1d():
    class Conv1D(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    class DummyConfig:
        model_type = "gpt2"

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig()
            self.c_attn = Conv1D()
            self.c_proj = Conv1D()
            self.embed = torch.nn.Embedding(4, 2)
            self.lm_head = torch.nn.Linear(2, 4)

    model = DummyModel()
    module_names = find_all_linear_modules(model, freeze_vision_tower=False)
    assert sorted(module_names) == ["c_attn", "c_proj"]
