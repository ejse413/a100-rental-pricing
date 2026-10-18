# a100 rental: Hourly Prices Compared, What Fits in 80 GB, and When Vultr's $2.40 Rate Is the Right Call

Renting an NVIDIA A100 80GB currently costs between **$1.09 and $5.03 per GPU-hour** on demand, and the spread comes entirely from the seller, since it's the same chip everywhere. Hyperscalers charge hyperscaler prices: AWS lists the 80GB card at $3.43/hr on p4de, Google Cloud at $5.03. Specialized GPU clouds and marketplaces fill out the bottom half of the range. Vultr rents the same card at **$2.40/GPU-hour**, which puts it in the middle of the market: about 30% cheaper than AWS, roughly half of Google Cloud, and about 36% above the market median of $1.76/hr from a mid-September audit of 30–40 providers.

This guide walks through the full price table from the latest published-rate sweep, the memory math that decides whether one card is enough for your model, whether the 40GB variant is ever worth it, SXM4 versus PCIe, and the specific situations where Vultr's rate makes sense versus the cheapest hosts on the board.

> **Quick pick:** for short and bursty jobs, a two-hour inference test, a weekend fine-tune — hourly on-demand is the right product. Vultr's case is a flat, published rate on a mainstream commercial cloud, no marketplace guesswork about which host you land on. 👉 [check Vultr's current A100 price](https://bit.ly/VulTR)

## A100 rental prices right now

On-demand rates for the A100 80GB ranged from $1.09 (Thunder Compute) to $5.03 (Google Cloud) per GPU-hour in the most recent weekly price review, published September 11, 2026. Every figure below comes from the provider's own pricing page, covers US regions, and excludes reserved or committed-use discounts.

| Provider | On-demand $/GPU-hr | 2-hour cost | Notes |
| --- | --- | --- | --- |
| Thunder Compute | $1.09 | $2.18 | Lowest tracked rate |
| Runpod | $1.59 | $3.18 |  |
| Vast.ai | $1.73 | $3.46 | Median of 5 verified US/CA hosts; priced as compute plus 100 GB storage, bandwidth billed separately |
| Crusoe Cloud | $2.00 | $4.00 |  |
| **Vultr** | **$2.40** | **$4.79** | The rental this guide books |
| Modal | $2.50 | $5.00 |  |
| CoreWeave | $2.70 | $5.40 | Per-GPU math on an 8×A100 node |
| Lambda | $2.79 | $5.58 |  |
| Paperspace | $3.18 | $6.36 |  |
| AWS EC2 (p4de.24xlarge) | $3.43 | $6.86 | 8-GPU instance; per-GPU normalization |
| Azure | $3.67 | $7.35 | Cheapest US region (eastus); ranges to $4.78 |
| Oracle Cloud | $4.00 | $8.00 |  |
| Google Cloud (a2-ultragpu-1g) | $5.03 | $10.06 | Cheapest US region; ranges to $6.04 |

A 100-hour training run makes the gaps concrete: $240 at Vultr, $200 at Crusoe, $343 at AWS, $503 at Google Cloud. Region matters too. Cloud rates swing 5–20% depending on where you deploy, and Azure's spread across its eight US A100 regions runs from $3.67 all the way to $4.78. Also read the fine print on marketplace rates: Vast.ai's figure bundles a 100 GB storage allocation while bandwidth gets billed separately, so the sticker price needs a second look before you compare it apples to apples.

Spot capacity undercuts all of this for fault-tolerant work, but it's a different product since the provider can reclaim the machine. Spheron, for example, listed spot A100 80GB at $1.12/hr alongside its $1.43 dedicated rate in late August. Plan checkpointed, restartable jobs on spot; pay on-demand rates for anything you need to stay up.

## Vultr's A100 rental: the rate and the bill

Vultr rents the A100 80GB at **$2.40 per GPU-hour** on demand, and billing follows usage, so every total is just hours multiplied by the rate:

| How you'll use it | The math | What you pay | Book it |
| --- | --- | --- | --- |
| Short test or demo (2 h) | $2.40 × 2 | ~$4.79 | [ Rent on Vultr](https://bit.ly/VulTR) |
| Training run (100 h) | $2.40 × 100 | $240 | [ Rent on Vultr](https://bit.ly/VulTR) |
| Fine-tune sprint (300 h) | $2.40 × 300 | $720 | [ Rent on Vultr](https://bit.ly/VulTR) |
| Pegged 24/7 for a month (730 h) | $2.40 × 730 | $1,752 | [ Rent on Vultr](https://bit.ly/VulTR) |

Two honest observations before you book. First, if you're pinning a card around the clock for months, hourly on-demand is the wrong product anywhere: past roughly **369 hours of use per month**, monthly dedicated terms win the math, and one dedicated-server outfit currently lists a bare-metal A100 at $649/month (about $0.89/h effective) against that $1.76/hr market median. Second, if the lowest possible sticker price is your only criterion, marketplace hosts undercut $2.40 — Thunder Compute tracked $1.09 and Vast.ai's verified hosts cluster in the $1.70s. Vultr's pitch is the middle path: a mainstream commercial cloud with a flat published rate, priced about 30% under AWS for identical silicon.

👉 [see Vultr's A100 rate and deploy](https://bit.ly/VulTR)

## What actually fits in 80 GB of A100 memory

Treat one A100 80GB as roughly **73.6 GiB of usable memory**: the card reports 81,920 MiB, vLLM claims 92% of GPU memory by default, and weights, KV cache and activations all share what's left. That number decides what you can rent one card for, and most price-comparison articles skip it entirely.

| Workload | Memory it needs | Fits on one A100 80GB? |
| --- | --- | --- |
| Llama-3.3-70B, 4-bit AWQ | 39.8 GB checkpoint | Yes, with ~120,000 tokens of BF16 KV cache left |
| 70B model, FP8 (weight-only on Ampere) | 72.7 GB checkpoint | Barely — short contexts only, ~19,000 tokens of KV cache |
| QLoRA fine-tune, 70B model | 41 GB (Unsloth) / 40–48 GB (Axolotl) | Yes |
| gpt-oss-120b, MXFP4 | 65.2 GB weights + 4.5 GiB KV per full 131k-token sequence | Yes, ~2.8 full-length sequences |
| Qwen3-32B, BF16 + one 32k sequence | ~69.0 GiB | Just inside the budget, little headroom |
| 16-bit LoRA, 32B model | 64–80 GB | Tight on one card; comfortable across two with FSDP |
| Llama-3.3-70B, BF16 | 141.1 GB | No — needs 4+ cards |
| Full fine-tune, 8B model | ~131 GB of model states | Only across 8×A100 with ZeRO-3 or FSDP |
| Full fine-tune, 70B model | 1,120–1,260 GB | No — 16+ cards, multi-node |

The takeaway: quantization is the lever that sets your bill. A 70B model at 4-bit fits on a single card with room for a six-figure context; the same model at BF16 turns into a four-card booking. If your workload matches a "yes" row, a single-GPU rental is all you need. Multi-GPU NVLink nodes only become necessary for BF16 70B, full fine-tunes, or long-context FP8 serving.

👉 [book a single A100 80GB on Vultr](https://bit.ly/VulTR)

## 80GB vs 40GB: rent the bigger card

Rent the 80GB in almost every case, because it often costs the same or less than the 40GB while doubling memory. The cheapest 40GB rate tracked is Vast.ai's median of $0.68 across 10 US/CA hosts; at list prices like Paperspace's $3.09 or Google Cloud's $3.67 for a 40GB card, an 80GB rental at Thunder Compute's $1.09 is cheaper outright. The extra memory buys larger batches, longer context windows, and headroom for bigger models without spilling to slow CPU offload.

There's also a serving trick the 40GB card can't match: MIG partitioning. One 80GB A100 splits into up to seven isolated 10GB instances, so a single rental can run several small inference services side by side. The 40GB only earns its place when it's priced well below the 80GB — that $0.68 marketplace rate for short jobs on small models — otherwise it's paying less for a worse deal.

## SXM4 or PCIe: does the form factor matter?

Only if you're training across multiple GPUs. The chip is identical in both versions, with the same published peak throughput; what changes is the power limit (400W vs 300W), a sliver of memory bandwidth (2,039 vs 1,935 GB/s), and the interconnect. SXM4 boards give every GPU 600 GB/s of NVLink through the HGX server board, while PCIe cards get a 2-GPU NVLink bridge and a 64 GB/s PCIe Gen4 host link.

Single-card work — QLoRA, 4-bit inference, classic computer vision — won't notice any of this, so book whichever form factor is cheapest. Multi-GPU training exchanges gradients between GPUs on every step, and 600 GB/s of NVLink versus 64 GB/s of PCIe Gen4 is close to a 10× gap. That's why 8×SXM4 nodes are the standard rental for PyTorch DDP, DeepSpeed ZeRO-3 and Megatron-LM runs; a verified red flag before starting one is running `nvidia-smi topo -m` and confirming NV entries actually appear between GPU pairs.

## A100 or H100: when the older card is still the right rental

For FP16 and TF32 work, the A100 does the same job as an H100 at **40–60% lower hourly cost**, which is why it hasn't left rental catalogs. The H100 is roughly 2.5–3× faster on tensor math and 1.7× on memory bandwidth, adds FP8 via its Transformer Engine, and costs about twice as much. The A100 has no FP8 — that arrived with Hopper — so FP8-native inference or training should pay for the H100. Everything in the 7B–30B fine-tuning range, quantized 70B serving, detection, segmentation, ASR and recommender workloads runs fine on Ampere.

For a sense of scale: the A100's TF32 throughput of 156 TFLOPS (dense) is about 10× a V100's FP32, with up to 5× faster BERT time-to-solution and roughly 2.6× average speedup measured across 23 networks. The Ampere stack is also the most mature in production; every framework and kernel has been tuned on it for years.

## A100 rental FAQ

### How much does it cost to rent an A100 per day?

Twenty-four hours times the hourly rate. At Vultr's $2.40 that's $57.60/day; the tracked range runs from $26.16 (Thunder Compute) to $120.72 (Google Cloud) for the same 80GB card.

### Can I fine-tune a 70B model on one rented A100?

With QLoRA, yes — Unsloth's 70B recipe needs about 41 GB and Axolotl's runs 40–48 GB, all inside a single 80GB card. Full fine-tuning the same model requires 1,120–1,260 GB of model states, which means 16 or more cards across multiple nodes. That's a cluster project, not a single-GPU reservation.

### Why do AWS, Azure and Google charge so much more?

List rates for the identical 80GB card run $3.43 on AWS p4de, $3.67 on Azure's cheapest US region, and $5.03 on Google Cloud, versus $1.09–$2.79 across the specialized GPU clouds in the table above. Regional pricing on the big clouds also moves 5–20%, so the gap widens or narrows depending on where you deploy.

### Is the A100 outdated?

Ampere lacks FP8, and that's the main reason to step up to an H100. For sub-70B training, INT8 inference, and everything living in TF32 or FP16, the A100 remains the value pick — it's still stocked by every provider in this guide at rates from $1.09 to $5.03, and it does fine-tuning, vision and speech work for about half the monthly cost of its successor.

## Before you book

- **Bursty fine-tuning, demos, nightly inference** → hourly on-demand. Floor pricing sits at Thunder Compute's $1.09; a mainstream commercial cloud at a mid-market rate is 👉 [rent an A100 80GB on Vultr](https://bit.ly/VulTR)
- **70B QLoRA or 4-bit serving** → one 80GB card from any provider in the table. Quantize first and skip the multi-GPU upsell.
- **BF16 70B, full fine-tunes, long-context FP8** → a multi-GPU SXM4 node with NVLink, budgeted as 8 cards, not one.
- **24/7 for months** → request monthly or dedicated terms instead; past ~369 hours a month, hourly rates stop making sense.

Every figure here comes from published pricing pages checked in September 2026, and rates move with supply — confirm the current number before committing to a long run.
