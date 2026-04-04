# Drift Protocol -- Pre-Hindsight Simulation
## Layer 3 Behavioral Detection Framework Applied Chronologically

**Rule: NO HINDSIGHT.** Each event is processed knowing only what has occurred up to that point. The system does not know the outcome.

---

## EVENT 1 -- ~March 11
**New wallet, mixer-adjacent funding, Solana**

### What the system observes
A previously unseen wallet receives a small SOL transfer. The funding source is linked to a Tornado Cash-adjacent mixer. No contract deployments, no token creation, no protocol interactions. Just a funded wallet.

### Behavioral classification
- **Funding trail:** Mixer-origin funding detected. This is the weakest possible signal -- mixers are used by privacy-conscious legitimate users, by MEV operators hiding strategies, and by malicious actors. The signal is non-zero but near-zero discrimination.
- **Velocity:** None. Single funding event.
- **Dormant state:** Wallet is newly created and idle. No stored potential yet.

### Stored potential assessment
One funded wallet. Potential energy: the SOL balance (small). No permissions granted, no contracts deployed, no positions taken. The wallet can do anything or nothing.

### Structural tension: 1/10
A mixer-funded wallet exists. This happens thousands of times per day across all chains. No trajectory to assess -- single point, no direction.

### Recommendation: IGNORE
Below monitoring threshold. Mixer-funded wallets are too common to track individually. Would only become relevant if later behavior escalates.

### What the system cannot know
- Who controls this wallet
- What the funds will be used for
- Whether the mixer usage is for privacy, evasion, or operational security
- Whether this wallet is connected to any other wallets (no on-chain link yet)

---

## EVENT 2 -- March 11-22 (~11 days)
**Token creation, liquidity pool, wash trading**

### What the system observes
The same wallet deploys an SPL token ("CarbonVote Token", 750M supply). Creates a Raydium pool with ~$500 SOL liquidity. Over 11 days, the deployer and 2-3 linked wallets execute repeated small buy/sell trades. Zero organic volume -- all trading is from the deployer cluster.

### Behavioral classification
- **Self-test pattern:** DETECTED. Deployer interacting with own contract. The 2-3 associated wallets are a sybil cluster performing wash trades. This maps directly to our self_test_traps detection: deployer creates contract, tests it, no external users.
- **Velocity:** Low and sustained. Consistent daily activity over 11 days, not a burst.
- **Funding trail:** Same mixer-origin wallet. Cluster of 2-3 wallets -- check if they share a funder.
- **Anomaly:** A token with $500 liquidity and zero organic volume for 11 days is not a functioning market. It is either abandoned or serving a non-market purpose.

### Stored potential assessment
- A token exists with a price (artificial, but a price)
- A Raydium pool exists with price history (manufactured, but history)
- The deployer has demonstrated the ability to maintain a stable price through wash trading
- **Key question the system cannot answer:** what is the stable price FOR? Maintaining an artificial price has no value unless something will later reference that price.

### Structural tension: 3/10
Wash-traded low-liquidity token with zero organic interest. This is common -- thousands of memecoins, rug pulls, and abandoned tokens look exactly like this. The 11-day sustained wash trading is slightly unusual (most rugs happen faster), but it is not inherently threatening.

### Recommendation: MONITOR (low priority)
Flag the deployer cluster. Record the wash trading pattern. Log that the token has manufactured price history with zero organic demand. This is a known pattern for pump-and-dump staging, but at $500 liquidity it is economically irrelevant.

### What the system cannot know
- Whether the price history will be referenced by an oracle
- Whether this token serves any purpose beyond the pool itself
- Whether the wash trading is building toward something or is just a failed project
- The connection (if any) to any DeFi protocol

---

## EVENT 3 -- March 23
**Durable nonce accounts linked to Security Council members**

### What the system observes
Four durable nonce accounts are created. Two are linked to addresses belonging to members of a major DeFi protocol's Security Council multisig.

### Behavioral classification
- **Approval accumulation (analogous):** Durable nonces are Solana's equivalent of stored permissions. A pre-signed transaction with a durable nonce can be submitted at any future time. This is permission energy.
- **Dormant state transition:** These nonce accounts are loaded but unused. They represent stored potential -- transactions that CAN execute but haven't yet.
- **Org attribution:** Two accounts are linked to named Security Council members. This is the first connection between this activity cluster and a specific protocol.

### Stored potential assessment
**Significant escalation.** Durable nonces linked to multisig members create the following stored potential:
- Pre-signed transactions can be submitted at any time without the signer being online
- If the multisig threshold is met by the nonce holders, governance actions can be executed without real-time consent from all members
- This is analogous to our approval_watchlist: permissions exist that haven't been exercised yet

### Structural tension: 5/10
This is the first event that creates non-trivial structural tension. Durable nonces for Security Council members are not inherently malicious -- they could be routine key rotation, backup signing mechanisms, or operational convenience. But they create a specific capability: pre-authorized governance execution. The tension is: why do these nonce accounts exist, and when will they be used?

### Recommendation: INVESTIGATE
The combination of:
1. Mixer-funded deployer (Event 1)
2. Manufactured price history (Event 2)
3. Durable nonces for governance members (Event 3)

does not yet form a coherent threat narrative. But the governance nonces are the first event that creates meaningful stored potential. An analyst should:
- Verify whether these nonces are routine (do all Security Council members have them?)
- Check whether the nonce creation timing correlates with any protocol governance proposal
- Determine if the nonce accounts were created by the Council members themselves or by a third party

### What the system cannot know
- Whether the nonce holders authorized the nonce creation
- What transactions are pre-signed in the nonces
- Whether the nonces are a standard operational procedure for this protocol
- The connection between the CVT token deployer (Events 1-2) and the nonce accounts (Event 3) -- unless they share funding, the link is invisible

---

## EVENT 4 -- March 23-27 (4 days)
**Price stabilization, oracle pickup, vault test interactions**

### What the system observes
- CVT price stabilizes at ~$1.00 via continued wash trading
- An oracle service begins reporting CVT price
- The original wallet executes small test deposits/withdrawals against a major protocol's vault contracts

### Behavioral classification
- **Self-test pattern:** ESCALATED. The wallet is now interacting with a real protocol's vaults, not just its own token. Small deposits/withdrawals = testing functionality, confirming access, verifying behavior. This is exactly our self_test_traps pattern at the protocol level.
- **Stored potential convergence:** Multiple independent trajectories are converging:
  - A token with artificial but oracle-reported price
  - Test access to protocol vaults
  - Pre-existing durable nonces for governance members
- **Anomaly (oracle):** An oracle reporting a price for a token with $500 liquidity and zero organic volume is a data integrity failure in the oracle, not a failure in our detection. But it means the artificial price now has downstream consequences -- any system that reads that oracle will treat CVT as a $1 token.

### Stored potential assessment
**Critical accumulation phase.** The stored potential is now:
- A token with an oracle-attested price (enables collateral usage)
- Vault access confirmed (test deposits succeeded)
- Governance nonces pre-signed (enables admin actions)

These three capabilities, if combined, enable: deposit worthless token as collateral (oracle says it has value) -> borrow real assets against it -> extract from vaults. But this is one of many possible interpretations. The system cannot reason about the exploit path -- it can only observe that multiple capabilities are converging around a single protocol.

### Structural tension: 7/10
The trajectory is now clear enough to assess: multiple independent capabilities are being assembled around a single protocol. Each capability alone is benign (tokens get oracle prices; wallets interact with vaults; governance members have nonces). The convergence of all four around the same protocol, from a mixer-funded deployer, is what elevates the tension.

### Recommendation: ALERT
Generate a convergence alert:
- "Multiple stored-potential vectors converging on [PROTOCOL]: artificial token with oracle price + vault test interactions + governance nonce accounts. Deployer cluster funded via mixer. No exploit detected. Structural tension elevated."
- Priority: HIGH
- Action: Notify protocol security team. Request verification of governance nonce legitimacy.

### What the system cannot know
- Whether the vault tests are reconnaissance or legitimate usage
- Whether the oracle listing is automatic or manipulated
- Whether someone at the protocol authorized these interactions
- The exploit mechanics (collateral manipulation is one theory, but so is governance capture, flash loan attack, etc.)

---

## EVENT 5 -- March 27
**Governance change: 3/5 -> 2/5 threshold, timelock removed, member swapped**

### What the system observes
The protocol's Security Council multisig executes three governance changes in one transaction or rapid sequence:
1. Signing threshold reduced from 3-of-5 to 2-of-5
2. Timelock on governance actions set to zero (immediate execution)
3. One council member replaced with a new member

### Behavioral classification
- **Anomaly: CRITICAL.** Three governance safety mechanisms weakened simultaneously:
  - Lower threshold = fewer signers needed = easier to execute
  - Zero timelock = no delay = no time for community to react
  - Member swap = potentially introduces an attacker-controlled signer
- **Structural tension resolution:** The durable nonces from Event 3 now have a clearer purpose. With a 2/5 threshold, two nonce holders can execute governance actions. The nonces were created BEFORE the threshold was lowered -- the sequence suggests the nonce creation anticipated the governance change.

### Stored potential assessment
**The safety margin has collapsed.** Before Event 5:
- An attacker needed 3 of 5 signers to act
- Any governance action had a timelock during which it could be vetoed
- All 5 council members were presumably vetted

After Event 5:
- Only 2 signers needed
- Actions execute immediately
- One signer is new and unvetted

Combined with the pre-existing durable nonces for 2 council members, the governance mechanism can now be exercised with pre-signed transactions at any time, with zero delay, with no veto window. This is maximum stored potential.

### Structural tension: 9/10
This is a categorical state change, not an incremental escalation. The protocol's governance has been reconfigured to enable rapid, low-threshold execution at exactly the moment when pre-signed transactions exist for governance members. The sequence is:
1. Create nonce accounts for council members (Mar 23)
2. Lower threshold to 2/5, remove timelock, swap member (Mar 27)
3. ???

The system does not know what step 3 is. But the trajectory is: capability assembly -> safety removal -> ???. In our framework, this is "stored potential approaching discharge."

### Recommendation: ALERT (CRITICAL)
- "CRITICAL: Protocol governance safety mechanisms degraded. Threshold 3/5 -> 2/5 + timelock removed + member swapped. Pre-existing durable nonce accounts for council members created March 23 (4 days before governance change). Deployer cluster with mixer funding and artificial oracle-priced token interacting with this protocol's vaults."
- Priority: CRITICAL
- Action: **Immediate escalation to protocol team.** Request justification for governance changes. Verify new council member identity. Verify durable nonce authorization.

### What the system cannot know
- Whether the governance change was voted on legitimately by the existing council
- Whether the new member is legitimate
- Whether the council members whose nonces exist authorized them
- Whether this is a routine security upgrade (some protocols do reduce thresholds temporarily for operational reasons) or an attack setup
- **The key uncertainty:** did the existing council members knowingly make these changes, or were their keys compromised?

---

## EVENT 6 -- March 28-30 (3 days)
**New nonce for new member, continued vault testing, CVT stable**

### What the system observes
- A new durable nonce account appears, linked to the NEW Security Council member (added March 27)
- Original wallet continues small test interactions with protocol vaults
- CVT token price remains stable near $1 with zero organic volume

### Behavioral classification
- **Stored potential: MAXIMUM.** The new council member now also has a durable nonce. With a 2/5 threshold, any combination of 2 nonce holders can execute governance actions via pre-signed transactions.
- **Self-test continuation:** Vault interactions continue. The deployer is still testing, not executing. This is a staging pattern.
- **Wash trading maintenance:** CVT price maintenance continues. 19 days of artificial price history now.

### Stored potential assessment
Every vector is loaded:
- Governance: 2/5 threshold + multiple nonce holders + zero timelock
- Oracle: artificial price accepted and reporting for 19 days
- Vault access: confirmed via repeated test interactions
- Funding: mixer-origin, clustered wallets

The only thing NOT yet observed is the discharge event -- the actual use of these capabilities.

### Structural tension: 9/10 (holding)
The tension has not increased from Event 5 -- it was already near maximum. But it has not resolved either. The system is observing a fully loaded state with no discharge. In our framework, a fully loaded state that holds for days is EITHER:
1. About to discharge (imminent exploit)
2. Abandoned (operator walked away)
3. Legitimate (these are all normal operations)

The system cannot distinguish these three states. It can only report that stored potential remains at maximum.

### Recommendation: ALERT (CRITICAL, repeat)
If the Event 5 alert was not acted upon, escalate. The addition of a nonce for the new council member strengthens the original alert. If protocol team has not responded, this becomes an external notification candidate (community, other security teams).

### What the system cannot know
- When the discharge will occur (could be hours, days, or never)
- What specific exploit path will be used
- Whether the protocol team received and dismissed the earlier alert as routine

---

## EVENT 7 -- April 1, 16:04 UTC
**Insurance fund test, then two rapid durable-nonce transactions**

### What the system observes
- Protocol executes a "routine" insurance fund test withdrawal
- 1 minute later: two transactions submitted in rapid succession (4 Solana slots apart)
- Both use durable nonces (pre-signed, not real-time signed)
- First transaction: modifies admin controls
- Second transaction: initiates large vault withdrawal

### Behavioral classification
- **Dormant activation: FIRING.** The durable nonces created March 23-30 are now being used. Stored potential is discharging.
- **Velocity: EXTREME.** Two critical governance actions in <1 second. Admin modification + vault withdrawal in sequence is a privilege escalation + extraction pattern.
- **Structural pattern:** insurance fund test immediately before the attack is a cover/distraction pattern -- or a final verification. Either way, the timing is non-coincidental.

### Stored potential assessment
**Discharge in progress.** All accumulated stored potential is converting to kinetic action:
- Governance nonces: USED (admin controls modified)
- Vault access: USED (withdrawal initiated)
- The only vector not yet observed in use: the CVT token / oracle price

### Structural tension: 10/10
This is the collapse event. The system should be in maximum alert state. The characteristics are:
- Pre-signed transactions (preparation, not improvisation)
- Admin modification BEFORE withdrawal (privilege escalation chain)
- Rapid sequential execution (automated, no human-in-loop delay)
- Pattern matches: dormant activation + velocity spike + stored permission discharge

### Recommendation: ALERT (CRITICAL, EMERGENCY)
- "EMERGENCY: Durable nonce transactions firing against [PROTOCOL]. Admin controls modified + vault withdrawal initiated. Pre-staged attack executing. This is not a drill."
- At this point, the alert is reactive -- the transactions are already submitted. The system's value was in Events 3-5, not Event 7.

### What the system cannot know
- Whether any defense is possible (transactions are already on-chain)
- Whether the protocol has circuit breakers that can halt withdrawals
- The total amount at risk

---

## EVENT 8 -- April 1, 16:05-16:20 UTC
**$285M extracted, laundering begins**

### What the system observes
- ~$285M flows from protocol vaults to a single wallet
- Immediate swapping via Jupiter aggregator
- Cross-chain bridging via deBridge and Wormhole to Ethereum
- Multiple Ethereum addresses converting to ETH

### Behavioral classification
- **Fund extraction: CONFIRMED EXPLOIT.** At this scale, this is unambiguously an exploit, not legitimate activity.
- **Laundry pipeline: ACTIVE.** Multi-hop, multi-chain, multi-address extraction pattern. This maps directly to our LAUNDRY_PIPELINE and CASHOUT_MOVEMENT alert types.
- **Bridge usage:** Cross-chain fund movement to complicate tracing. Matches our bridge_events monitoring pattern.

### Stored potential assessment
Fully discharged. All accumulated potential has converted to extracted value. The remaining "stored potential" is in the laundering pipeline -- funds in transit that haven't reached final destination.

### Structural tension: 0/10 (post-event)
The tension has resolved. The exploit is complete. The system transitions from detection to forensics.

### Recommendation: TRACK (forensic mode)
- Track all destination addresses across chains
- Map the full laundering pipeline
- Flag all addresses receiving funds for exchange/bridge monitoring
- Generate a complete fund flow graph

### What the system cannot know
- Final destination of funds
- Identity of the attacker
- Whether additional attacks are planned using the same infrastructure
- Whether the CVT token served a role in the exploit mechanics (oracle manipulation for collateral) or was a separate operation

---

# Post-Simulation Analysis

## A. First Meaningful Alert

**Event 3 (March 23)** -- when durable nonce accounts linked to Security Council members were created. This was the first event that created specific, non-trivial stored potential against an identifiable target. Events 1-2 were too noisy to act on.

However, Event 3 alone would have been a MONITOR, not an ALERT. The first alert-worthy signal was **Event 4** (March 23-27), when the convergence of oracle-priced token + vault test interactions + governance nonces created enough structural tension to justify contacting the protocol.

The first CRITICAL alert was **Event 5 (March 27)**, when governance safety mechanisms were degraded. At this point, the system had a clear, articulable basis for emergency escalation.

## B. Maximum Pre-Exploit Warning Time

- Event 3 (first nonce creation): March 23 -- **9 days before execution**
- Event 4 (convergence alert): March 23-27 -- **5-9 days before execution**
- Event 5 (CRITICAL alert): March 27 -- **5 days before execution**
- Event 6 (repeated CRITICAL): March 28-30 -- **2-4 days before execution**

**Maximum actionable warning: 5 days** (from Event 5 CRITICAL alert to Event 7 execution).

If the system had generated a MONITOR at Event 3 and an analyst had connected it to Events 1-2, the theoretical maximum is 9 days. But this requires connecting mixer-funded token creation to governance nonce creation -- a link that may not be visible on-chain.

## C. Strongest Signals (ranked)

1. **Governance threshold reduction + timelock removal** (Event 5) -- the single strongest signal. Three safety mechanisms weakened simultaneously is extremely unusual and has few legitimate explanations.

2. **Durable nonce creation for governance members** (Event 3) -- stored potential for governance execution. Strong signal, but only in context of later events.

3. **Pre-signed durable nonce transactions executing in rapid sequence** (Event 7) -- unmistakable at the moment of execution, but too late for prevention.

4. **Convergence of multiple capabilities around one protocol** (Events 1-4 combined) -- the pattern of token + oracle + vault access + governance nonces is more significant than any individual event.

5. **Wash trading with zero organic volume for 11+ days** (Event 2) -- a staging indicator, but common enough to be noisy alone.

## D. Weakest / Most Ambiguous Signals

1. **Mixer-funded wallet creation** (Event 1) -- thousands per day, near-zero discrimination
2. **New token creation with small liquidity** (Event 2) -- extremely common, especially on Solana
3. **Small vault test transactions** (Events 4, 6) -- indistinguishable from legitimate user testing
4. **Governance member changes** (Event 5, taken alone) -- protocols change council members for legitimate reasons

## E. False Positive Exposure

- **Event 1 (mixer funding):** Hundreds of thousands of mixer-funded wallets exist. FP rate: >99.9%
- **Event 2 (token + wash trading):** Thousands of new tokens with manufactured volume launch weekly. FP rate: >99%
- **Event 3 (durable nonces):** Durable nonces are a standard Solana feature. Many multisig operations use them. FP rate: ~95%
- **Event 4 (vault testing):** Millions of wallet-to-protocol interactions happen daily. FP rate: >99.99%
- **Event 5 (governance changes):** This is where false positives drop significantly. Simultaneous threshold reduction + timelock removal + member swap is rare. Estimated FP rate: <5% of governance changes have this pattern. But it is not zero -- emergency operational changes do happen.
- **Events 5+3 combined (governance change + pre-existing nonces):** Very low FP rate. Pre-creating nonces days before lowering the threshold to enable those nonces is a narrow behavioral pattern. Estimated FP rate: <1%

## F. What Was Truly Invisible

1. **Key compromise** -- if council members' private keys were stolen (rather than the members being complicit), this is completely invisible on-chain. The nonce creation and governance changes look identical whether authorized or unauthorized.

2. **Off-chain coordination** -- any communication between the attacker and compromised/complicit council members happened off-chain. The system sees only the on-chain artifacts.

3. **The exploit mechanism** -- until Event 7, the system could not know HOW the attack would execute. Oracle manipulation? Direct vault drain? Governance takeover? All were possible.

4. **The CVT token's role** -- whether CVT was used as fake collateral, as a price manipulation tool, or was a separate operation entirely is unclear from on-chain data alone until the exploit transaction reveals the mechanics.

5. **Attribution** -- nothing in the on-chain data identifies who is behind the mixer-funded wallet. The connection to any nation-state or criminal organization is invisible to behavioral surveillance.

## G. Would an Analyst Have Escalated at Event 5?

**Yes, if they saw Event 5 in context of Events 3-4. No, if they saw Event 5 in isolation.**

Event 5 alone -- governance threshold lowered, timelock removed, member swapped -- could be explained as "the protocol is doing an emergency operational change." Protocols do this. It's unusual but not unheard of.

But Event 5 in context of:
- Durable nonces created 4 days earlier for the members who can now act with the lowered threshold
- A mixer-funded deployer testing this protocol's vaults
- An artificial token with oracle-reported price interacting with this ecosystem

This context transforms Event 5 from "unusual governance change" to "the final safety mechanism being removed before a pre-staged attack." An analyst who had visibility into Events 3-4 would have escalated. An analyst who only saw Event 5 would have noted it and waited.

**This is the core argument for temporal behavioral surveillance.** No single event in this sequence is conclusive. The signal is in the trajectory -- the accumulation of stored potential, the convergence of capabilities, and the removal of safety mechanisms in a specific sequence. A snapshot tool seeing Event 5 would see a governance change. A behavioral system tracking Events 1-6 would see an attack being assembled.

The 5-day window between Event 5 and Event 7 was enough time to freeze the governance, verify council member keys, and potentially prevent $285 million in losses. The signal existed. The question is whether anyone was watching the right trajectory.
