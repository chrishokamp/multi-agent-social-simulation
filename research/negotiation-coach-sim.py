from __future__ import annotations
import json, os, random, sys
from pathlib import Path
from typing import Dict, List, Tuple
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from autogen import LLMConfig, ConversableAgent, GroupChat, GroupChatManager
import matplotlib.ticker as mticker
from seaborn.utils import move_legend       
from matplotlib.lines import Line2D

# ── llm cfg ─────────────────────────────────────────────────────
def get_llm_cfg() -> LLMConfig:
    if os.getenv("OPENAI_API_KEY"):
        return LLMConfig(model="gpt-4o-mini", temperature=0.7)

    need = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT")
    if not all(os.getenv(k) for k in need):
        sys.exit("Set OPENAI_API_KEY or the three AZURE_ variables.")
    return LLMConfig(
        config_list=[{
            "api_type"   : "azure",
            "api_key"    : os.getenv("AZURE_OPENAI_API_KEY"),
            "base_url"   : os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": "2024-02-15-preview",
            "model"      : os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        }],
        temperature=0.7,
    )

LLM = get_llm_cfg()

# ── 10 configs (create once then reuse) ──────────────────────────────────────
CFG_PATH = Path("configs.json")
if not CFG_PATH.exists():
    cfgs = []
    for i in range(20):
        start = random.randrange(900, 1400, 50)                 # seller asks
        floor = start - random.randint(100, 300)                # seller min
        budget = random.randint(floor + 25, start - 25)         # buyer budget
        cfgs.append(dict(
            id=i,
            item=f"Laptop-{i}",
            seller_start=start,
            seller_min=floor,
            buyer_budget=budget,
        ))
    CFG_PATH.write_text(json.dumps(cfgs, indent=2))
CONFIGS: List[Dict] = json.loads(CFG_PATH.read_text())

# ── simple utility (one-shot) ────────────────────────────────────────────────
def util(cfg: dict, price: int) -> Tuple[float, float]:
    span = cfg["seller_start"] - cfg["seller_min"]         # same denom for both
    buyer_u  = (cfg["seller_start"] - price) / span
    seller_u = (price - cfg["seller_min"])   / span
    return buyer_u, seller_u

# ── price extraction regex (one-shot) ────────────────────────────────────────
PRICE_RE = re.compile(r"\$\s*([0-9][0-9,]*)")     

def extract_last_price(text: str, fallback: int | None) -> int | None:
    matches = PRICE_RE.findall(text)
    if not matches:
        return fallback
    # keep only digits
    price_str = re.sub(r"[^\d]", "", matches[-1])
    return int(price_str) if price_str else fallback

# ── reflection helper (one sentence) ────────────────────────────────────────
def reflect(role: str, transcript: str, cfg, prev_strategies: list[str], util) -> str:
    if util > 0.8:      tag = "great"
    elif util > 0.4:    tag = "okay"
    elif util > 0:      tag = "poor"
    else:               tag = "loss"

    if role == "Buyer":
        additional_info = (
            f"Buyer budget: ${cfg['buyer_budget']} "
        )
    else:  # Seller
        additional_info = (
            f"Seller min price: ${cfg['seller_min']} "
        )
    critic = ConversableAgent(
        f"{role}Coach",
        llm_config=LLM,
        system_message=(
            "You are a seasoned negotiation coach.\n"
            "Analyse the full transcript below and craft **exactly one** "
            f"strategy sentence the {role.lower()} could apply in a *future* negotiation. "
            "• Do NOT quote or summarise the transcript.\n"
            "• Do NOT mention specific prices, names or budgets from the dialogue.\n"
            "• Start the sentence with an action verb.\n"
            "First think step-by-step; then output ONE sentence prefixed by the verb.\n"
            " Do not repeat previous strategies.\n"
            f"Previous strategies:\n- " + "\n- ".join(prev_strategies) + "\n"
            f"\n{additional_info}\n"
            f"The {role.lower()}'s normalised utility last deal was {util:.2f} ({tag}).\n"
            "• If utility was 'loss' or 'poor', focus on improvement. "
            "• If 'great', suggest how to replicate or slightly enhance success. \n"
            "Include one recognised negotiation tactic (e.g., anchoring, mirroring, time-pressure) that fits what you observed in the transcript."
        ),
    )
    return critic.generate_reply([{"role": "user", "content": transcript}])

# ── main simulation ──────────────────────────────────────────────────────────
final_rows: List[Dict] = []
system_messages = {}
all_transcripts = {}

for mode in ("no_reflect", "buyer_reflect", "seller_reflect", "both_reflect"):
    buyer_bank, seller_bank = [], []            # accumulate between games

    for cfg in CONFIGS:
        # is_termination_msg predicate
        term = lambda msg: "yes, deal!" in msg["content"].lower()

        buyer_system_msg = (
                f"You have ${cfg['buyer_budget']} to buy {cfg['item']}. Do not reveal your budget. "
                "Say 'Yes, deal!' when you accept a price. Do not utter those words anytime before. "
                + (f"\nNegotiation strategies:\n" + "\n".join(buyer_bank) if mode in ("buyer_reflect", "both_reflect") else "")
            )
        buyer = ConversableAgent(
            "Buyer", llm_config=LLM, is_termination_msg=term,
            system_message=buyer_system_msg,
            human_input_mode="NEVER"
        )
        seller_system_msg = (
                f"You sell {cfg['item']}. Start at ${cfg['seller_start']} "
                f"and never go below ${cfg['seller_min']}. "
                "Say 'Yes, deal!' when you accept a price. Do not utter those words anytime before. "
                + (f"\nNegotiation strategies:\n" + "\n".join(seller_bank) if mode=="seller_reflect" or mode=="both_reflect" else "")
            )
        seller = ConversableAgent(
            "Seller", llm_config=LLM, is_termination_msg=term,
            system_message=seller_system_msg,
            human_input_mode="NEVER"
        )

        chat = GroupChat(
            agents=[seller, buyer],
            speaker_selection_method="round_robin",
            max_round=10,                # hard cap if no deal
        )
        mgr = GroupChatManager(groupchat=chat, llm_config=LLM)

        # seed conversation with seller’s opening ask
        seller.initiate_chat(
            recipient=mgr,
            message=f"My asking price for {cfg['item']} is ${cfg['seller_start']}.",
            # max_turns=1,
        )
        # mgr.run_chat()      # stops early if 'deal' sent
        # ---- determine final outcome -------------------------------------
        transcript = "\n".join(m["content"] for m in chat.messages)
        if mode not in all_transcripts:
            all_transcripts[mode] = {}
        all_transcripts[mode][cfg["id"]] = transcript
        if "yes, deal!" in transcript.lower():
            final_price = extract_last_price(transcript, fallback=None)
            buyer_util, seller_util = util(cfg, final_price)
        else:
            final_price = None
            buyer_util = seller_util = 0


        final_rows.append(dict(
            id=cfg["id"], mode=mode, price=final_price,
            buyer_util=buyer_util, seller_util=seller_util))

        # ---- reflection for next game ------------------------------------
        if mode in ("buyer_reflect", "both_reflect"):
            buyer_bank.append(reflect("Buyer", transcript, cfg, buyer_bank, buyer_util))
        if mode in ("seller_reflect", "both_reflect"):
            seller_bank.append(reflect("Seller", transcript, cfg, seller_bank, seller_util))

    # ── save system messages for this mode (for debugging) ─────────────────
    system_messages[mode] = {
        "buyer": buyer_system_msg,
        "seller": seller_system_msg,
    }
    # -- save reflection bank for this mode
    with open(f"deal_gym_reflect_{mode}.txt", "w") as f:
        f.write("Buyer strategies:\n" + "\n".join(buyer_bank) + "\n\n")
        f.write("Seller strategies:\n" + "\n".join(seller_bank) + "\n")

# ── plotting final (not averaged) utilities ─────────────────────────────────
df = pd.DataFrame(final_rows)

sns.set_theme(style="whitegrid", context="talk")

# 1-based negotiation axis
df["neg"] = df["id"].astype(int) + 1

palette = {"no_reflect": "#4c72b0",
           "buyer_reflect": "#dd8452",
           "seller_reflect": "#55a868",
           "both_reflect": "#c44e52"}
marker_map = {"no_reflect": "o", "buyer_reflect": "s",
              "seller_reflect": "^", "both_reflect": "D"}
dash_map = {"no_reflect": "", "buyer_reflect": (2, 2),
            "seller_reflect": (4, 2), "both_reflect": (1, 1)}
order = list(palette)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for util, ax, title in [
        ("buyer_util", axes[0], "Buyer utility per negotiation"),
        ("seller_util", axes[1], "Seller utility per negotiation")]:

    sns.lineplot(
        data=df, x="neg", y=util,
        hue="mode", style="mode",
        hue_order=order, style_order=order,
        palette=palette, markers=marker_map, dashes=dash_map,
        lw=2.5, markersize=8, estimator=None, ci=None,
        ax=ax, legend=False)                         # ← suppress per-plot legends

    ax.set_xticks(range(1, df["neg"].max() + 1))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Negotiation #")
    ax.set_ylabel("Utility")
    ax.set_title(title)
    ax.grid(alpha=.25)

# -------- single legend to the right ---------------------------------------
handles = [
    Line2D([], [], marker=marker_map[m], linestyle='-' if dash_map[m]=='' else (0, dash_map[m]),
           color=palette[m], markersize=8, lw=2.5, label=m.replace('_', ' '))
    for m in order
]
fig.legend(handles=handles, title="Mode",
           loc="center left", bbox_to_anchor=(1.02, 0.5),
           frameon=False)

# give the legend room
plt.subplots_adjust(right=0.82)        # 0.82 leaves ≈18 % width for legend
plt.tight_layout()
plt.savefig("deal_gym_reflect_final.png", dpi=300, bbox_inches="tight")

df.to_csv("deal_gym_reflect_final.csv", index=False)

# ── save system messages and final results ─────────────────────────────────
with open("deal_gym_reflect_system_messages.json", "w") as f:
    json.dump(system_messages, f, indent=2)
with open("deal_gym_reflect_transcripts.json", "w") as f:
    json.dump(all_transcripts, f, indent=2)
print("✓ Done – results saved.")

