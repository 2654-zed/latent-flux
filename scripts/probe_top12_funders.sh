#!/bin/bash
# Eth-trace each of the top 12 funders. Output JSON per address.
set -e
mkdir -p scripts/funder_traces

# Top 12 funders (excluding 0xf70da978 which is already done)
FUNDERS=(
    "0xfd92f4e91d54b9ef91cc3f97c011a6af0c2a7eda"
    "0x3304e22ddaa22bcdc5fca2269b418046ae7b566a"
    "0xc43f317ed4d81cbbfe2c9c98b4cc6f303519f078"
    "0xb0b0b6903489cc56bf037cb2f5ba986e2775bb07"
    "0xde8eb937cb5475eee5ac96dce6ba2d18e439c473"
    "0x0e6e91775d24d34b90e0f3d806a90705f0199999"
    "0x238d7170f309a55b87a144a341bd6105897082ca"
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1"
    "0x8ca702323c341a8d46ee94a2abeddb08798ca10d"
    "0x39591e7c099a379fd7b349ebfecaeef439c40454"
    "0xca7ece5e43ef44de8e430629a5b535eca48e251b"
)

for f in "${FUNDERS[@]}"; do
    echo "Probing $f..."
    railway ssh --service latent-flux "python -m surveillance.eth_depth --address $f --hops 1" \
        > "scripts/funder_traces/${f}.json" 2>&1
    echo "  saved to scripts/funder_traces/${f}.json"
done
echo "DONE — 11 traces saved (0xf70da978 already exists at scripts/eth_trace_f70da978.json)"
