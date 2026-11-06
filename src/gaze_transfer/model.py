"""Compact-VLM adapter with training-only KL and observed-gaze ablations."""
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ['pretrained', 'label_only', 'attention_kl', 'static', 'temporal', 'shuffled',
         'caption_transfer', 'structured_no_gaze', 'structured', 'structured_state', 'allocated']
GAZE_MODES = {'static', 'temporal', 'shuffled', 'caption_transfer', 'structured', 'structured_state', 'allocated'}
TRANSFER_MODES = {'caption_transfer', 'structured_no_gaze', 'structured', 'structured_state', 'allocated'}


class GazeAttention(nn.Module):
    """Single-sequence minibatches; all ablations instantiate identical capacity."""
    def __init__(self, dimension, hidden=64):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.query = nn.Linear(dimension, hidden, bias=False)
        self.key = nn.Linear(dimension, hidden, bias=False)
        self.value = nn.Linear(dimension, hidden, bias=False)
        self.out = nn.Linear(hidden, dimension, bias=False)
        self.gaze_encoder = nn.GRU(5, hidden, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.01))
        self.gaze = None
        self.mode = 'label_only'
        self.attention = None
        self.keep_indices = None

    def forward(self, tokens):
        # (frames, spatial tokens, language hidden); batch size is one window.
        if self.mode == 'pretrained' and self.keep_indices is None:
            return tokens
        x = self.norm(tokens)
        q = self.query(x.mean((0, 1), keepdim=True))
        if self.mode in GAZE_MODES and self.gaze is not None:
            gaze = self.gaze.to(device=tokens.device, dtype=tokens.dtype)
            valid = gaze[:, 3] > 0
            if valid.any():
                sequence = gaze[valid].unsqueeze(0)
                if self.mode == 'static':
                    sequence = sequence.mean(1, keepdim=True)
                    sequence = sequence.clone(); sequence[:, :, 0] = 0
                _, h = self.gaze_encoder(sequence)
                q = q + h[-1, :, None, :]
        logits = (q * self.key(x)).sum(-1) / math.sqrt(q.shape[-1])
        self.attention = logits.float().softmax(-1)
        context = (self.attention.to(x.dtype)[..., None] * self.value(x)).sum(1, keepdim=True)
        result = tokens + self.gate.tanh() * self.out(context)
        if self.keep_indices is not None:
            index = self.keep_indices.to(result.device)
            result = result.gather(1, index[:, :, None].expand(-1, -1, result.shape[-1]))
        return result

    def kl(self, heatmap, valid):
        if self.attention is None:
            raise RuntimeError('Adapter has not processed images')
        valid = valid.to(self.attention.device)
        if not valid.any():
            return self.attention.sum() * 0
        target = heatmap.to(self.attention.device).float()
        if target.shape != self.attention.shape:
            raise ValueError(f'Gaze heatmap/token mismatch: {target.shape}, {self.attention.shape}')
        return F.kl_div(self.attention[valid].clamp_min(1e-9).log(), target[valid], reduction='batchmean')


class CompactVLM(nn.Module):
    def __init__(self, model_path, device='cuda', lora=True):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        # One 512px image view per frame; record actual processor token counts.
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.size = {'longest_edge': 512}
        raw = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            attn_implementation='sdpa')
        raw.requires_grad_(False)
        if lora:
            from peft import LoraConfig, get_peft_model
            self.vlm = get_peft_model(raw, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=r'.*text_model.*\.(q_proj|v_proj)', task_type='CAUSAL_LM'))
        else:
            self.vlm = raw
        self.adapter = GazeAttention(raw.config.text_config.hidden_size)
        self.adapter.to(dtype=next(raw.parameters()).dtype)
        self.hook = raw.model.connector.register_forward_hook(lambda module, args, output: self.adapter(output))
        self.raw = raw  # shared module, never separately saved as a full checkpoint
        self.keep_tokens = 0
        self.spatial_maps = None
        self.to(device)

    def set_context(self, gaze, mode, spatial_maps=None, keep_tokens=0):
        if mode not in MODES:
            raise ValueError(mode)
        self.adapter.gaze = gaze
        self.adapter.mode = mode
        self.spatial_maps, self.keep_tokens = spatial_maps, keep_tokens
        self.adapter.keep_indices = None

    def inputs(self, images, question, answer=None):
        content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': question}]
        prompt = self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                     tokenize=False, add_generation_prompt=True)
        encoded = self.processor(text=prompt, images=images, return_tensors='pt')
        if answer is not None:
            text = prompt + answer + self.processor.tokenizer.eos_token
            whole = self.processor(text=text, images=images, return_tensors='pt')
            labels = whole['input_ids'].clone()
            labels[:, :encoded['input_ids'].shape[1]] = -100
            whole['labels'] = labels
            encoded = whole
        if self.keep_tokens:
            ids = encoded['input_ids'][0]
            positions = (ids == self.raw.config.image_token_id).nonzero().flatten()
            if positions.numel() % len(images):
                raise ValueError('Unexpected processor image-token layout')
            original = positions.numel() // len(images)
            keep = min(self.keep_tokens, original)
            uniform = torch.linspace(0, original-1, keep).round().long()
            indices = []
            for frame in range(len(images)):
                if self.adapter.mode in GAZE_MODES and self.spatial_maps is not None:
                    # Retain peripheral coverage as well as highest gaze-density patches.
                    context = torch.linspace(0, original-1, max(1, keep//2)).round().long().tolist()
                    ranked = torch.as_tensor(self.spatial_maps[frame]).argsort(descending=True).tolist()
                    selected = context + [i for i in ranked if i not in context][:keep-len(context)]
                    indices.append(sorted(selected))
                else:
                    indices.append(uniform.tolist())
            self.adapter.keep_indices = torch.tensor(indices, dtype=torch.long)
            retained = torch.ones(len(ids), dtype=torch.bool)
            retained[positions] = False
            for frame, selected in enumerate(indices):
                retained[positions[frame*original+torch.tensor(selected)]] = True
            for key in ['input_ids', 'attention_mask', 'labels']:
                if key in encoded:
                    encoded[key] = encoded[key][:, retained]
        device = next(self.adapter.parameters()).device
        dtype = next(self.raw.parameters()).dtype
        return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                for k, v in encoded.items()}

    def forward(self, inputs):
        return self.vlm(**inputs, use_cache=False)

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=96):
        output = self.vlm.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                   use_cache=True)
        return self.processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def score_answers(self, images, question, answers):
        scores = []
        for answer in answers:
            encoded = self.inputs(images, question, answer)
            output = self(encoded)
            labels = encoded['labels'][:, 1:]
            valid = labels != -100
            logp = output.logits[:, :-1].float().log_softmax(-1)
            values = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            scores.append(float(values[valid].sum()))
        return torch.tensor(scores).softmax(0).tolist()

    def save(self, destination):
        from pathlib import Path
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
        self.vlm.save_pretrained(destination/'lora')
        torch.save(self.adapter.state_dict(), destination/'gaze_adapter.pt')

    def load(self, destination):
        from pathlib import Path
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        destination = Path(destination)
        set_peft_model_state_dict(self.vlm, load_file(str(destination/'lora/adapter_model.safetensors')))
        self.adapter.load_state_dict(torch.load(destination/'gaze_adapter.pt', map_location='cpu', weights_only=True))
