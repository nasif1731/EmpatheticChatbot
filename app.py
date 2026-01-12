# --- put this at the very TOP of app.py ---
import os
os.environ["TORCH_DISABLE_TORCHDYNAMO"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# Point the app to your saved weights + tokenizer
os.environ["ED_MODEL_PATH"] = os.environ.get("ED_MODEL_PATH", "best_transformer_model.pt")
os.environ["ED_SPM_PATH"]   = os.environ.get("ED_SPM_PATH",   "spm_bpe.model")

# (optional) keep these consistent with your training config
os.environ["ED_MAXLEN"]  = os.environ.get("ED_MAXLEN",  "256")   # seqLen in your notebook
os.environ["ED_DMODEL"]  = os.environ.get("ED_DMODEL",  "512")   # embed_dim
os.environ["ED_HEADS"]   = os.environ.get("ED_HEADS",   "2")     # num_heads
os.environ["ED_ENC"]     = os.environ.get("ED_ENC",     "2")     # encoder_layers
os.environ["ED_DEC"]     = os.environ.get("ED_DEC",     "2")     # decoder_layers
os.environ["ED_DROPOUT"] = os.environ.get("ED_DROPOUT", "0.3")   # dropout
# --- end snippet ---

# app.py
import re, io, math, unicodedata, torch, gradio as gr, sentencepiece as spm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------
# Special token IDs (exactly as training)
# ----------------------
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ----------------------
# Normalization (exactly as training)
# ----------------------
def normalize(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s).lower()
    s = (s.replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'")
         .replace("—","-").replace("–","-").replace("…","..."))
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------
# Helper masks
# ----------------------
def make_causal_mask(size):
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)

# ----------------------
# >>> MODEL (names/structure MATCH your training code) <<<
# with optional attention storing on decoder cross-attn (no params -> no load issues)
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, store_last_attn: bool = False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # projection names match training
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.store_last_attn = store_last_attn
        self.last_attn = None  # (N, H, Lq, Lk)

    def forward(self, query, key, value, mask=None):
        N, Lq, d_model = query.shape
        Lk = key.shape[1]
        Q = self.q_proj(query).view(N, Lq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(N, Lk, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(N, Lk, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attention_weights = self.dropout(F.softmax(scores, dim=-1))
        if self.store_last_attn:
            self.last_attn = attention_weights.detach()
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(N, Lq, d_model)
        return self.out_proj(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # names match training
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_pad_mask: torch.BoolTensor) -> torch.Tensor:
        attn_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)
        norm_src = self.attn_norm(src)
        attn_out = self.self_attn(norm_src, norm_src, norm_src, mask=attn_mask)
        src = src + self.attn_dropout(attn_out)
        norm_src = self.ffn_norm(src)
        ffn_out = self.ffn(norm_src)
        src = src + self.ffn_dropout(ffn_out)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, store_cross_attn=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, store_last_attn=store_cross_attn)
        self.attn1_norm = nn.LayerNorm(d_model)
        self.attn2_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.attn1_dropout = nn.Dropout(dropout)
        self.attn2_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_pad_mask: torch.BoolTensor, memory_pad_mask: torch.BoolTensor) -> torch.Tensor:
        L_tgt = tgt.size(1)
        causal_mask = make_causal_mask(L_tgt).to(tgt.device)
        tgt_attn_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)
        full_tgt_mask = tgt_attn_pad_mask | causal_mask
        norm_tgt = self.attn1_norm(tgt)
        attn_out1 = self.self_attn(norm_tgt, norm_tgt, norm_tgt, mask=full_tgt_mask)
        tgt = tgt + self.attn1_dropout(attn_out1)
        mem_attn_mask = memory_pad_mask.unsqueeze(1).unsqueeze(2)
        norm_tgt = self.attn2_norm(tgt)
        attn_out2 = self.cross_attn(query=norm_tgt, key=memory, value=memory, mask=mem_attn_mask)
        tgt = tgt + self.attn2_dropout(attn_out2)
        norm_tgt = self.ffn_norm(tgt)
        ffn_out = self.ffn(norm_tgt)
        tgt = tgt + self.ffn_dropout(ffn_out)
        return tgt

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_pad_mask: torch.BoolTensor) -> torch.Tensor:
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_pad_mask)
        return self.norm(src)

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1, max_len: int = 5000, store_cross_attn=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, store_cross_attn=(i==num_layers-1 and store_cross_attn)) for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_pad_mask: torch.BoolTensor, memory_pad_mask: torch.BoolTensor) -> torch.Tensor:
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_decoder(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_pad_mask, memory_pad_mask)
        return self.norm(tgt)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=2, d_ff=2048, num_encoder_layers=2, num_decoder_layers=2, dropout=0.3, max_len=256, num_emotions=None, store_cross_attn=False):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_decoder_layers, dropout, max_len, store_cross_attn=store_cross_attn)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.emo_head = nn.Linear(d_model, num_emotions) if num_emotions is not None else None
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # tie weights
        self.generator.weight = self.decoder.embedding.weight

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor, src_pad: torch.BoolTensor, tgt_pad: torch.BoolTensor):
        memory = self.encoder(src, src_pad)
        output = self.decoder(tgt_in, memory, tgt_pad, src_pad)
        logits = self.generator(output)
        emo_logits = None
        if self.emo_head is not None:
            mask = (~src_pad).float().unsqueeze(-1)
            pooled = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            emo_logits = self.emo_head(pooled)
        return logits, emo_logits

# ----------------------
# Token helpers (same as training)
# ----------------------
def detok(ids, sp, pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    keep=[]
    for t in ids:
        if t==pad_id or t==bos_id: 
            continue
        if t==eos_id: 
            break
        keep.append(t)
    return sp.decode(keep)

# ----------------------
# Build inputs (exactly like your training)
# ----------------------
def build_input_text(emotion, situation, user_text, use_emo_tokens=True):
    e  = normalize(emotion)   if emotion   else ""
    si = normalize(situation) if situation else ""
    ut = normalize(user_text) if user_text else ""
    if use_emo_tokens and e:
        emo_tag = f"<emo_{re.sub(r'[^a-z0-9_]+','_',e)}>"
        x = f"{emo_tag} <sep> {si} <sep> {ut} Agent:"
    else:
        x = f"Emotion: {e} | Situation: {si} | Customer: {ut} Agent:"
    return re.sub(r"\s+"," ",x).strip()

# ----------------------
# Decoding (same logic as your notebook, plus light degen controls)
# ----------------------
def _violates_no_repeat_ngram(tokens, n):
    if n <= 0: return False
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    if len(tokens) < 2*n: return False
    last = tokens[-n:]
    for i in range(len(tokens)-n):
        if tokens[i:i+n] == last:
            return True
    return False

@torch.no_grad()
def greedy_decode(model, sp, src_ids, max_new_tokens=64, no_repeat_ngram_size=3, min_len=6):
    m = model
    m.eval()
    dev = next(m.parameters()).device
    src = src_ids.to(dev).unsqueeze(0)
    src_pad = (src == PAD_ID)
    tgt = torch.full((1,1), BOS_ID, dtype=torch.long, device=dev)
    for step in range(max_new_tokens):
        tgt_pad = (tgt == PAD_ID)
        out = m(src, tgt, src_pad, tgt_pad)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :].squeeze(0)
        if step < max(min_len, 1):
            logits[EOS_ID] = -1e9
        nxt = torch.argmax(logits).view(1,1)
        cand = torch.cat([tgt.squeeze(0), nxt.view(-1)], dim=0)
        if no_repeat_ngram_size>0 and _violates_no_repeat_ngram(cand, no_repeat_ngram_size):
            logits[nxt] = -1e9
            nxt = torch.argmax(logits).view(1,1)
        tgt = torch.cat([tgt, nxt], dim=1)
        if int(nxt.item()) == EOS_ID: break
    return detok(tgt.squeeze(0), sp)

@torch.no_grad()
def beam_search_decode(model, sp, src_ids, num_beams=5, max_new_tokens=64, length_penalty=0.7, no_repeat_ngram_size=3, min_len=6):
    m = model
    m.eval()
    dev = next(m.parameters()).device
    src = src_ids.to(dev).unsqueeze(0)
    src_pad = (src == PAD_ID)
    beams = [(torch.tensor([[BOS_ID]], device=dev, dtype=torch.long), 0.0, False)]
    for step in range(max_new_tokens):
        new_beams=[]
        for tokens, logp, finished in beams:
            if finished:
                new_beams.append((tokens, logp, True)); continue
            tgt_pad = (tokens == PAD_ID)
            out = m(src, tokens, src_pad, tgt_pad)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[:, -1, :].squeeze(0)
            if step < max(min_len, 1):
                logits[EOS_ID] = -1e9
            probs = torch.log_softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, k=num_beams)
            for k in range(topi.size(0)):
                nxt = topi[k].view(1,1)
                cand = torch.cat([tokens, nxt], dim=1)
                if no_repeat_ngram_size>0 and _violates_no_repeat_ngram(cand.squeeze(0), no_repeat_ngram_size):
                    continue
                lp = logp + float(topv[k].item())
                fin = nxt.item() == EOS_ID
                new_beams.append((cand, lp, fin))
        if not new_beams: break
        def score(entry):
            t, lp, fin = entry
            L = max(1, t.size(1)-1)
            denom = ((5+L)**length_penalty)/((5+1)**length_penalty)
            return lp/denom
        new_beams.sort(key=score, reverse=True)
        beams = new_beams[:num_beams]
        if all(b[2] for b in beams): break
    best = max(beams, key=lambda x: x[1])[0]
    return detok(best.squeeze(0), sp)

@torch.no_grad()
def sample_decode(model, sp, src_ids, max_new_tokens=64, temperature=0.9, top_k=40, top_p=0.9, no_repeat_ngram_size=3, min_len=6):
    def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
        logits = logits.clone()
        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[logits < kth] = -float("inf")
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs > top_p
            mask[..., 0] = False
            sorted_logits[mask] = -float("inf")
            logits = torch.full_like(logits, -1e9)
            logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        return logits

    m = model
    m.eval()
    dev = next(m.parameters()).device
    src = src_ids.to(dev).unsqueeze(0)
    src_pad = (src == PAD_ID)
    tgt = torch.tensor([[BOS_ID]], device=dev, dtype=torch.long)
    for step in range(max_new_tokens):
        tgt_pad = (tgt == PAD_ID)
        out = m(src, tgt, src_pad, tgt_pad)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :].squeeze(0)
        if step < max(min_len, 1):
            logits[EOS_ID] = -1e9
        if temperature != 1.0:
            logits = logits / max(1e-6, temperature)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        if torch.isneginf(logits).all():
            nxt = logits.argmax().view(1,1)
        else:
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).view(1,1)
        cand = torch.cat([tgt.squeeze(0), nxt.view(-1)], dim=0)
        if no_repeat_ngram_size>0 and _violates_no_repeat_ngram(cand, no_repeat_ngram_size):
            logits[int(nxt.item())] = -1e9
            alt = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(alt, 1).view(1,1)
        tgt = torch.cat([tgt, nxt], dim=1)
        if int(nxt.item()) == EOS_ID: break
    return detok(tgt.squeeze(0), sp)

# ----------------------
# Load assets (STRICT key match, including emo_head size)
# ----------------------
def load_assets(model_path="best_transformer_model.pt",
                spm_path="spm_bpe.model",
                d_model=512, heads=2, enc_layers=2, dec_layers=2,
                dropout=0.3, max_len=256, store_cross_attn=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(); sp.load(spm_path)
    vocab = sp.get_piece_size()
    print(f"[spm] vocab size = {vocab}")
    print("[spm] specials[0..3] =", [sp.id_to_piece(i) for i in range(4)])

    # discover emotion tokens from tokenizer (training did this over train X)
    emo_pieces = []
    for i in range(vocab):
        p = sp.id_to_piece(i)
        if re.match(r"^<emo_[a-z0-9_]+>$", p):
            emo_pieces.append(p)
    emo_list_clean = sorted({re.match(r"^<emo_([a-z0-9_]+)>$", t).group(1) for t in emo_pieces})
    # training used: emotions = ["unknown"] + emo_tags
    num_emotions = 1 + len(emo_list_clean) if len(emo_list_clean) > 0 else None

    model = Transformer(
        src_vocab_size=vocab,
        tgt_vocab_size=vocab,
        d_model=d_model,
        num_heads=heads,
        d_ff=4*d_model,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dropout=dropout,
        max_len=max_len,
        num_emotions=num_emotions,           # IMPORTANT: match training if emo_head was present
        store_cross_attn=store_cross_attn
    ).to(device)

    sd = torch.load(model_path, map_location=device)
    # strict=True because names match exactly
    model.load_state_dict(sd, strict=True)
    print("[load] strict OK")

    model.eval()
    return model, sp, device, emo_list_clean

model, sp, device, EMO_LIST = load_assets(
    model_path=os.environ["ED_MODEL_PATH"],
    spm_path=os.environ["ED_SPM_PATH"],
    d_model=int(os.environ["ED_DMODEL"]),
    heads=int(os.environ["ED_HEADS"]),
    enc_layers=int(os.environ["ED_ENC"]),
    dec_layers=int(os.environ["ED_DEC"]),
    dropout=float(os.environ["ED_DROPOUT"]),
    max_len=int(os.environ["ED_MAXLEN"]),
    store_cross_attn=True
)

# ----------------------
# Attention heatmap util (last decoder cross-attn, averaged over heads)
# ----------------------
def last_cross_attention_avg():
    try:
        a = model.decoder.layers[-1].cross_attn.last_attn
    except Exception:
        a = None
    if a is None:
        return None
    a = a.mean(dim=1)  # (N, Lq, Lk)
    return a[0].detach().cpu().numpy()

def make_heatmap(attn_mat, src_ids, sp, title="Cross-attention (last step)"):
    if attn_mat is None or attn_mat.size == 0:
        return None
    last_row = attn_mat[-1]  # focus on last generated token
    toks = []
    for t in src_ids.tolist():
        if t==PAD_ID: continue
        toks.append(sp.id_to_piece(int(t)))
    toks = toks[:len(last_row)]
    fig = plt.figure()
    plt.imshow(last_row[np.newaxis, :], aspect="auto")
    plt.yticks([0],["last token"])
    plt.xticks(range(len(toks)), toks, rotation=90)
    plt.title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as _Image
    return _Image.open(buf).convert("RGB")

# ----------------------
# Gradio interaction
# ----------------------
def respond(user_text, emotion, situation, strategy, max_new_tokens, beam_size, length_penalty,
            temperature, top_k, top_p, no_repeat_ng, show_attn, history):

    if history is None:
        history = []

    # Build model input exactly like training
    x = build_input_text(emotion, situation, user_text, use_emo_tokens=True if EMO_LIST else False)
    src_ids = torch.tensor(sp.encode(x, out_type=int)[:int(os.environ["ED_MAXLEN"])],
                           dtype=torch.long, device=device)

    # Choose decoding
    if strategy=="Greedy":
        reply = greedy_decode(model, sp, src_ids, max_new_tokens=max_new_tokens, no_repeat_ngram_size=no_repeat_ng)
    elif strategy=="Beam":
        reply = beam_search_decode(model, sp, src_ids, num_beams=beam_size, max_new_tokens=max_new_tokens,
                                   length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ng)
    else:
        reply = sample_decode(model, sp, src_ids, max_new_tokens=max_new_tokens, temperature=temperature,
                              top_k=top_k, top_p=top_p, no_repeat_ngram_size=no_repeat_ng)

    # Heatmap
    heatmap_img = None
    if show_attn:
        attn = last_cross_attention_avg()
        heatmap_img = make_heatmap(attn, src_ids, sp)

    # OpenAI-style messages for Chatbot(type="messages")
    user_msg = f"[{emotion}] {user_text}" if emotion else user_text
    messages = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    return messages, messages, heatmap_img

with gr.Blocks(title="Empathetic Chatbot (Transformer)") as demo:
    gr.Markdown("# Empathetic Chatbot")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=420)
            user_in = gr.Textbox(label="Your message", lines=3, placeholder="Type your message...")
            emotion = gr.Dropdown(choices=EMO_LIST if EMO_LIST else None, value=None, allow_custom_value=True,
                                  label="Emotion (optional)")
            situation = gr.Textbox(label="Situation (optional)", placeholder="Short context/situation")
            with gr.Accordion("Decoding", open=False):
                strategy = gr.Radio(choices=["Greedy","Beam","Top-p"], value="Beam", label="Strategy")
                max_new = gr.Slider(8, 128, value=64, step=1, label="Max new tokens")
                beam_sz = gr.Slider(2, 10, value=5, step=1, label="Beam size")
                len_pen = gr.Slider(0.5, 1.4, value=0.7, step=0.05, label="Length penalty")
                temp = gr.Slider(0.2, 1.5, value=0.9, step=0.05, label="Temperature")
                topk = gr.Slider(0, 200, value=40, step=1, label="Top-k")
                topp = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p")
                no_rep = gr.Slider(0, 6, value=3, step=1, label="No-repeat n-gram size")
            show_attn = gr.Checkbox(value=False, label="Show attention heatmap (last step)")
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=1):
            gr.Markdown("### Model")
            gr.Markdown(f"Device: **{torch.device('cuda' if torch.cuda.is_available() else 'cpu').type}**")
            gr.Markdown(f"Vocab: **{sp.get_piece_size()}**")
            if EMO_LIST:
                gr.Markdown(f"Emotions: {', '.join(EMO_LIST[:12])}" + (" ..." if len(EMO_LIST)>12 else ""))
            else:
                gr.Markdown("Emotions: using free-text")
            attn_img = gr.Image(type="pil", label="Attention heatmap", visible=True)

    state = gr.State([])

    send.click(
        respond,
        inputs=[user_in, emotion, situation, strategy, max_new, beam_sz, len_pen,
                temp, topk, topp, no_rep, show_attn, state],
        outputs=[chatbot, state, attn_img]
    )
    clear.click(lambda: ([], [], None), None, [chatbot, state, attn_img])
    user_in.submit(
        respond,
        inputs=[user_in, emotion, situation, strategy, max_new, beam_sz, len_pen,
                temp, topk, topp, no_rep, show_attn, state],
        outputs=[chatbot, state, attn_img]
    )

if __name__ == "__main__":
    demo.launch(share=True)
