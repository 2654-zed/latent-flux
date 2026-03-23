# BOT INVESTIGATION: 0x84792c2a
**Classification:** CUSTOM SPRAY BOT — UNPROFITABLE
**Generated:** 2026-03-22 23:45 UTC

---

## Summary
An EOA running a proprietary bot framework that sends ~5-10 reverted transactions per block across Arbitrum, hitting unknown contracts with a unique function selector (`2f139e4f`) that no other bot in the 19,149-contract corpus uses. It has burned $4,412 in gas over 51 days with zero visible revenue. The bot operates 24/7 with a human-supervised refueling pattern and appears to be a custom-built arbitrage or sniping bot that has never successfully extracted value.

---

## Target Profile
| Field | Value |
|---|---|
| Total reverts captured | 5,133 (in 751 block events) |
| Revert per block | 5-10 (consistent, never varies) |
| TX events on monitored contracts | **0** — it hits contracts OUTSIDE our corpus |
| Contracts hit | Unknown — bot_candidate_events tracks reverts per block, not per contract |
| Known org overlap | **Cannot determine** — its targets aren't in our contracts table |
| Assessment | Hitting contracts NOT flagged by our surveillance system |

**Critical finding:** This bot has ZERO entries in `transaction_events`. The revert_cluster_detector catches it via full-block scanning (counting reverted txs per address per block), but it never touches any of our 19,149 monitored contracts. Its targets are entirely outside our detection corpus — legitimate protocols, unflagged contracts, or infrastructure we haven't catalogued.

---

## Selector Analysis
| Field | Value |
|---|---|
| Selector | `2f139e4f` |
| Known function | **NOT IN 4byte.directory / NOT TAGGED** |
| Other bots using this selector | **0 — COMPLETELY UNIQUE** |
| Total calls with this selector | 5,102 (from this bot only) |
| Assessment | **Custom-built proprietary bot function** |

**This is the most diagnostic finding.** Out of 451 tracked bots, 1,742 selector entries, and 19,149 contracts, not a single other address has ever used `2f139e4f`. This isn't a purchased bot kit or shared framework. Someone wrote custom smart contract interaction logic and deployed it exclusively on this one EOA.

This eliminates the "cheap purchased bot" hypothesis. Someone invested development effort to build this, then burned $4,412 running it for nothing.

---

## Timing Pattern

### Hourly Heatmap (UTC)
```
Hour:  00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23
Revs:  ## ## #  ##  .  .  .  ## #   .  #  #  ## ## ## ## ## ##  # ## ## ##  #  .
```

**Peak hours: 13:00-15:00 UTC and 19:00-20:00 UTC.**
**Quiet hours: 04:00-06:00 UTC and 22:00-23:00 UTC.**

This is NOT 24/7 uniform — there's a clear daily rhythm. The quiet period (22:00-06:00 UTC, 8 hours) suggests either:
- The operator is in **UTC+0 to UTC+3** (Europe/Africa) — quiet during sleeping hours
- The bot targets a DEX with lower activity overnight
- The funding runs low and the bot pauses until refueled

### Daily Volume
| Date | Reverts | Blocks | Avg/Block |
|---|---|---|---|
| Mar 17 | 8 | 1 | 8.0 |
| Mar 18 | **1,580** | 231 | 6.8 |
| Mar 19 | **1,471** | 221 | 6.7 |
| Mar 20 | **1,081** | 156 | 6.9 |
| Mar 21 | **121** | 16 | 7.6 |
| Mar 22 | **872** | 126 | 6.9 |

**Mar 21 drop to 121 reverts** is notable — the bot nearly stopped for a day. Either it ran out of gas (refueled), or the operator paused it. The consistent ~7 reverts per block across ALL days means the bot's behavior is constant when it's running — the variation is only in how many hours per day it operates.

### Event Interval
| Bucket | Count | % |
|---|---|---|
| <1 second | 2 | 0.4% |
| 1-10 seconds | 64 | 12.8% |
| 10-60 seconds | 106 | 21.2% |
| 1-5 minutes | 186 | 37.3% |
| 5-60 minutes | 133 | 26.7% |
| >1 hour | 8 | 1.6% |

Average interval: 6.7 minutes between blocks where it reverts. This bot hits approximately one block every 7 minutes, spraying 5-10 transactions per attempt.

---

## Behavioral Classification

| Pattern | Finding |
|---|---|
| Transaction pattern | **Spray** — 5-10 txs per block, all reverted |
| Interval regularity | **Semi-regular** — ~7 min gaps, not clock-precise |
| Target switching | **Unknown** — targets outside our corpus |
| Revert consistency | **Perfectly consistent** — always 5-10 per block, never 1, never 20 |
| Event-driven | **Likely** — variable intervals suggest reacting to on-chain events |

The 5-10 reverts per block pattern is the signature. This bot sends a **burst of 5-10 transactions in the same block**, all to the same selector, all reverting. This is consistent with:
1. **Multi-path arbitrage attempts** — trying 5-10 different routing paths in one block, all fail
2. **Token sniping** — sending multiple buy attempts at slightly different parameters, all rejected
3. **Sandwich attempts** — trying to front-run/back-run with multiple gas levels, all outcompeted

---

## Financial Summary
| Field | Value |
|---|---|
| Total ETH received | 2.152 ETH (~$4,519) |
| Total ETH sent out | 0.045 ETH (~$95) |
| Current balance | 0.006 ETH (~$12) |
| **Gas burned** | **2.101 ETH (~$4,412)** |
| ERC-20 revenue | **$0** |
| Internal tx revenue | **$0** |
| Daily burn rate | ~$86/day |
| Days until empty | **~1 day** |
| Nonce | 372,474 |

### Funding Pattern
| Funder | Amount | Frequency | Pattern |
|---|---|---|---|
| `0x9ebed688...` | 0.1 ETH per tx | Every 1-2 weeks | Primary gas supplier |
| `0x260790b1...` | 0.005-0.049 ETH | Irregular | Secondary + receives small withdrawals |
| `0xb371b557...` | 0.009 ETH | One-time | Minor contributor |

The funder `0x9ebed688` keeps this bot alive with periodic 0.1 ETH top-ups. `0x260790b1` appears to be the operator's personal wallet — it both funds and receives small amounts back. Neither funder appears anywhere else in our surveillance database (not a deployer, not a bot, not in any org).

---

## Assessment

**Most likely explanation:** This is a custom-built arbitrage or token sniping bot that was developed by a solo operator with enough technical skill to write a proprietary smart contract function (`2f139e4f`) but not enough to make it profitable. The bot sprays 5-10 transactions per block at targets outside our trap corpus (likely legitimate DEX contracts or new token pairs), fails on every attempt, and has been doing so for 51 days while the operator periodically refuels it with 0.1 ETH.

The operator is likely in UTC+0 to UTC+3 (European timezone) based on the daily activity cycle. They check on the bot occasionally (March 21's low volume suggests they paused and restarted it), but they haven't looked at the economics — or they believe the bot will eventually "find" profitable opportunities and are running it as a long-tail experiment.

**Confidence:** MEDIUM

**What would change this assessment:**
- If `2f139e4f` matches a known DEX function not yet in 4byte.directory (could be a legitimate protocol interaction, not arb)
- If the bot's targets (outside our corpus) include any profitable interactions we can't see from the surveillance DB
- If `0x9ebed688` (the funder) funds other bots that ARE profitable — this bot might be one node in a larger operation where other nodes succeed
- If the reverted txs contain value transfers (not just gas) — some MEV strategies intentionally revert to block competing transactions

---

## Recommendations
- [ ] Decode `2f139e4f` via Alchemy `debug_traceTransaction` on one of its recent reverted txs — see what contract it's calling and why it reverts
- [ ] Monitor funding wallet `0x9ebed688` for links to other bot operations
- [ ] Track if bot goes offline when gas runs out (~1 day from now) and whether it gets refueled
- [ ] Add `2f139e4f` to known_selectors as `proprietary_spray_bot`
- [ ] Check if the bot's targets (the contracts it calls) overlap with any known DeFi protocol addresses
