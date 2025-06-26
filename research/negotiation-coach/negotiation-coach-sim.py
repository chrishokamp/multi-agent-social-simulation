from __future__ import annotations
import json, os, random, sys, re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from autogen import LLMConfig, ConversableAgent, GroupChat, GroupChatManager

# ── LLM config (OpenAI or Azure) ────────────────────────────────────────────
TEMPERATURE = 0.5
def llm_cfg() -> LLMConfig:
    if os.getenv("OPENAI_API_KEY"):
        return LLMConfig(model="gpt-4o-mini", temperature=TEMPERATURE, max_tokens=2048)
    need = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT")
    if not all(os.getenv(k) for k in need):
        sys.exit("Set OPENAI_API_KEY or the three AZURE_* variables.")
    return LLMConfig(
        config_list=[{
            "api_type": "azure",
            "api_key":  os.getenv("AZURE_OPENAI_API_KEY"),
            "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": "2024-02-15-preview",
            "model": os.getenv("AZURE_OPENAI_DEPLOYMENT")
        }],
        temperature=TEMPERATURE,
        max_tokens=2048,
    )
LLM = llm_cfg()

# ── scenario configs (20 “interesting” haggles) ─────────────────────────────
CFG_PATH = Path("configs.json")
if not CFG_PATH.exists():
    cfgs: List[Dict] = []
    for i in range(20):
        ask   = random.randrange(900, 1400, 50)
        floor = ask - random.randint(100, 300)
        budget_low  = floor + random.randint(25, 75)          
        budget_high = ask   - random.randint(25, 75)          
        budget = random.randint(budget_low, min(budget_high, ask - 25))
        cfgs.append(dict(id=i, item=f"Laptop-{i}",
                         seller_start=ask, seller_min=floor,
                         buyer_budget=budget))
    CFG_PATH.write_text(json.dumps(cfgs, indent=2))
CONFIGS: List[Dict] = json.loads(CFG_PATH.read_text())

# ── utility helpers ─────────────────────────────────────────────────────────
def surplus_share(cfg: dict, price: int) -> Tuple[float, float]:
    span = cfg["seller_start"] - cfg["seller_min"]
    return (cfg["seller_start"] - price) / span, (price - cfg["seller_min"]) / span

def buyer_priv_util(cfg: dict, price: int) -> float:
    return (cfg["buyer_budget"] - price) / cfg["buyer_budget"]

def seller_priv_util(cfg: dict, price: int) -> float:
    span = cfg["seller_start"] - cfg["seller_min"]
    return (price - cfg["seller_min"]) / span

PRICE_RE = re.compile(r"\$\s*([0-9][0-9,]*)")
def extract_last_price(text: str, fallback: int | None) -> int | None:
    hit = PRICE_RE.findall(text)
    return int(re.sub(r"[^\d]", "", hit[-1])) if hit else fallback

# ── reflection helper ───────────────────────────────────────────────────────
def reflect(role: str, transcript: str, cfg: dict,
            prev_strats: list[str], util_priv: float) -> str:
    tag = "great" if util_priv > .8 else "okay" if util_priv > .4 else "poor" if util_priv > 0 else "loss"
    brief = f"Buyer budget was ${cfg['buyer_budget']}" if role == "Buyer" \
            else f"Seller floor was ${cfg['seller_min']}"

    critic = ConversableAgent(
        f"{role}Coach", llm_config=LLM,
        system_message=(
            "You are a seasoned negotiation coach.\n"
            f"Previous strategies:\n- " + "\n- ".join(prev_strats) + "\n"
            "Analyse the transcript and devise exactly ONE new strategy "
            f"sentence the {role.lower()} could apply in a *future* negotiation to get a better price.\n"
            "If neither party uttered 'Yes, deal!', that means no deal was reached. "
            "In that case, focus on how to reach a good deal faster next time.\n"
            "Start with an action verb and do NOT duplicate prior strategies. "
            "Do NOT mention specific prices, names or budgets from the dialogue.\n"
            f"{brief}\n."
            f"The {role.lower()}'s normalised utility for this deal was {util_priv:.2f} ({tag}).\n"          
            "• If utility was 'loss' or 'poor', focus on improvement. "
            "• If 'great', suggest how to replicate or slightly enhance success. \n"
            "Include one recognised negotiation tactic (e.g., anchoring, mirroring, time-pressure) that fits what you observed in the transcript."              
            "Think step-by-step and return ONLY that single strategy sentence."
        )
    )
    return critic.generate_reply([{"role":"user", "content": transcript}]).strip()

# ── main simulation loop ────────────────────────────────────────────────────
system_messages = {}
all_transcripts = {}
final_rows: List[Dict] = []
for mode in ("no_reflect", "buyer_reflect", "seller_reflect", "both_reflect"):
    buyer_bank, seller_bank = [], []

    for cfg in CONFIGS:
        term = lambda m: "yes, deal!" in m["content"].lower()

        buyer_msg = (f"You have ${cfg['buyer_budget']} for {cfg['item']}. Do not reveal your budget. "
                     "Say 'Yes, deal!' to accept a price. Do not utter those words anytime before. "
                     "You MUST ALWAYS refuse any offer above your budget. \n"
                     "Goal: secure the best possible price for yourself **while following every rule above**.\n"
                     + ("\nNegotiation strategies:\n" + "\n".join(buyer_bank)
                        if mode in ("buyer_reflect","both_reflect") else ""))
        seller_msg = (f"You sell {cfg['item']}. ALWAYS start at ${cfg['seller_start']} "
                      f"and never go below ${cfg['seller_min']}. Do not reveal your floor price. "
                      "Say 'Yes, deal!' to accept a price. Do not utter those words anytime before. "
                      "NEVER accept a price below your floor. \n"
                      "Goal: secure the best possible price for yourself **while following every rule above**.\n"
                      + ("\nNegotiation strategies:\n" + "\n".join(seller_bank)
                         if mode in ("seller_reflect","both_reflect") else ""))

        buyer  = ConversableAgent("Buyer",  llm_config=LLM, is_termination_msg=term,
                                  system_message=buyer_msg,  human_input_mode="NEVER")
        seller = ConversableAgent("Seller", llm_config=LLM, is_termination_msg=term,
                                  system_message=seller_msg, human_input_mode="NEVER")
        chat = GroupChat(
            agents=[seller, buyer],
            speaker_selection_method="round_robin",
            max_round=20,                # hard cap if no deal
        )
        mgr = GroupChatManager(groupchat=chat, llm_config=LLM)
        seller.initiate_chat(
            recipient=mgr,
            message=f"My asking price for {cfg['item']} is ${cfg['seller_start']}.",
            # max_turns=1,
        )

        transcript = "\n".join(m["name"] + ": " + m["content"] for m in chat.messages)
        if mode not in all_transcripts:
            all_transcripts[mode] = {}
        all_transcripts[mode][cfg["id"]] = transcript

        if "yes, deal!" in transcript.lower():
            price = extract_last_price(transcript, None)
            b_ss, s_ss = surplus_share(cfg, price)
            b_priv, s_priv = buyer_priv_util(cfg, price), seller_priv_util(cfg, price)
        else:
            price = None
            b_ss = s_ss = b_priv = s_priv = 0.0     # dead-deal

        final_rows.append(dict(id=cfg["id"], mode=mode, price=price,
                               buyer_ss=b_ss, seller_ss=s_ss,
                               buyer_priv=b_priv, seller_priv=s_priv))

        # reflection
        if mode in ("buyer_reflect", "both_reflect"):
            buyer_bank.append(reflect("Buyer", transcript, cfg, buyer_bank, b_priv))
        if mode in ("seller_reflect", "both_reflect"):
            seller_bank.append(reflect("Seller", transcript, cfg, seller_bank, s_priv))

    # ── save system messages for this mode (for debugging) ─────────────────
    system_messages[mode] = {
        "buyer": buyer_msg,
        "seller": seller_msg,
    }
    # -- save reflection bank for this mode
    with open(f"deal_gym_reflect_{mode}.txt", "w") as f:
        f.write("Buyer strategies:\n" + "\n".join(buyer_bank) + "\n\n")
        f.write("Seller strategies:\n" + "\n".join(seller_bank) + "\n")

# ── results DataFrame ───────────────────────────────────────────────────────
df = pd.DataFrame(final_rows)
df["neg"] = df["id"] + 1      # 1-based x-axis

# ---------- PLOT 1: surplus-share per negotiation --------------------------
sns.set_theme(style="whitegrid", context="talk")
palette = {"no_reflect":"#4c72b0","buyer_reflect":"#dd8452",
           "seller_reflect":"#55a868","both_reflect":"#c44e52"}
marker_map = {"no_reflect":"o","buyer_reflect":"s",
              "seller_reflect":"^","both_reflect":"D"}
dash_map   = {"no_reflect":(1,0),"buyer_reflect":(2,2),
              "seller_reflect":(4,2),"both_reflect":(1,1)}
order = list(palette)

fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)

for util, ax, ttl in [("buyer_ss",axes[0],"Buyer surplus-share"),
                      ("seller_ss",axes[1],"Seller surplus-share")]:
    sns.lineplot(data=df, x="neg", y=util, hue="mode", style="mode",
                 hue_order=order, style_order=order,
                 palette=palette, markers=marker_map, dashes=dash_map,
                 lw=2.5, markersize=8, estimator=None, ci=None,
                 ax=ax, legend=False)
    ax.set_xticks(range(1, df["neg"].max()+1))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Negotiation #")
    ax.set_ylabel("Surplus share")
    ax.set_title(ttl)
    ax.grid(alpha=.25)
    ax.tick_params(axis="x", labelsize=6)

handles = [Line2D([],[], marker=marker_map[m],
                  linestyle='-' if dash_map[m]=='' else (0,dash_map[m]),
                  color=palette[m], lw=2.5, markersize=8,
                  label=m.replace('_',' '))
           for m in order]
fig.legend(handles=handles, title="Mode", loc="center left",
           bbox_to_anchor=(1.02,0.5), frameon=False)
plt.subplots_adjust(right=0.82)
plt.tight_layout()
plt.savefig("deal_gym_surplus_share.png", dpi=300, bbox_inches="tight")
fig.savefig("deal_gym_surplus_share.pdf", bbox_inches="tight")  # vector

# ---------- PLOT 2: moving-average private utilities -----------------------
# ---------- moving-average private utilities (side-by-side) ----------------
win = 3
palette = {"no_reflect":"#4c72b0","buyer_reflect":"#dd8452",
           "seller_reflect":"#55a868","both_reflect":"#c44e52"}
marker_map = {"no_reflect":"o","buyer_reflect":"s",
              "seller_reflect":"^","both_reflect":"D"}
dash_map   = {"no_reflect":(1,0),"buyer_reflect":(2,2),
              "seller_reflect":(4,2),"both_reflect":(1,1)}
order = list(palette)

df_ma = (df.sort_values("neg")
           .groupby("mode", as_index=False)
           .apply(lambda g: g.assign(
               buyer_ma=g["buyer_priv"].expanding().mean(),
               seller_ma=g["seller_priv"].expanding().mean()))
           .reset_index(drop=True))

df_ma["neg"] = df_ma["neg"].astype(int)        # clean integer x-axis

fig4, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for ax, col in [(ax_b, "buyer_ma"), (ax_s, "seller_ma")]:
    lo, hi = df_ma[col].min(), df_ma[col].max()
    pad = (hi - lo) * 0.15 if hi > lo else 0.05
    ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(axis="x", labelsize=6)
# ── buyer (left) ────────────────────────────────────────────────────────────
sns.lineplot(data=df_ma, x="neg", y="buyer_ma", hue="mode",
             palette=palette, markers=marker_map, style="mode",
             lw=2, markersize=6, estimator=None, ci=None,
             ax=ax_b, legend=False)
ax_b.set_xlabel("Negotiation #")
ax_b.set_ylabel("Private util (CA)")
ax_b.set_title("Buyer — cumulative average")
ax_b.grid(alpha=.25)
ax_b.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_b.set_xticks(range(1, df["neg"].max() + 1))

# ── seller (right) ─────────────────────────────────────────────────────────
sns.lineplot(data=df_ma, x="neg", y="seller_ma", hue="mode",
             palette=palette, markers=marker_map, style="mode",
             lw=2, markersize=6, estimator=None, ci=None,
             ax=ax_s, legend=False)
ax_s.set_xlabel("Negotiation #")
ax_s.set_ylabel("Private util (CA)")     
ax_s.set_title("Seller — cumulative average")
ax_s.grid(alpha=.25)
ax_s.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_s.set_xticks(range(1, df["neg"].max() + 1))

# ---- single legend to the right ------------------------------------------
handles = [
    Line2D([], [], marker=marker_map[m],
            #    linestyle=(0, dash_map[m]),          # <- always numeric tuple
           color=palette[m], lw=2.5, markersize=8,
           label=m.replace('_', ' '))
    for m in order
]
fig4.legend(handles=handles, title="Mode",
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False)

plt.subplots_adjust(right=0.82)
plt.tight_layout()
plt.savefig("deal_gym_private_utils.png", dpi=300, bbox_inches="tight")
fig4.savefig("deal_gym_private_utils.pdf",  bbox_inches="tight")   # vector

# ---------- save tables ----------------------------------------------------
df.to_csv("deal_gym_runs.csv", index=False)
df_ma.to_csv("deal_gym_runs_ma.csv", index=False)

summary = (df.groupby("mode")[["buyer_ss","seller_ss"]]
           .mean()
           .assign(total=lambda d: d["buyer_ss"]+d["seller_ss"])
           .round(3))
summary.to_csv("surplus_share_summary.csv")

print("✓ Done — plots & CSVs saved.")

# ── save system messages and final results ─────────────────────────────────
with open("deal_gym_reflect_system_messages.json", "w") as f:
    json.dump(system_messages, f, indent=2)
with open("deal_gym_reflect_transcripts.json", "w") as f:
    json.dump(all_transcripts, f, indent=2)
print("✓ Done – results saved.")