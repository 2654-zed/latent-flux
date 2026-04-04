"""
Entry point functions for multiprocessing.Process workers.

Must be in a separate importable module (not __main__) for 'spawn' start method.
"""

import asyncio
import logging


def monitor_entry(chain, rpc_url, write_queue):
    """Entry point for chain monitor subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{chain}:%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from surveillance.bytecode_classifier import classify
    from surveillance.deployment_monitor import run_monitor
    asyncio.run(run_monitor(
        rpc_url, None, classifier=classify, chain=chain,
        write_queue=write_queue,
    ))


def routing_entry(api_key, write_queue):
    """Entry point for routing monitor subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [routing:%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from surveillance.routing_monitor import run_routing_monitor
    asyncio.run(run_routing_monitor(api_key, None, write_queue=write_queue))
