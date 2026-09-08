#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print SpecSLO's Ascend graph, attention, tree and MC2 capability matrix."""

import argparse
import json

from vllm_ascend.spec_decode.pearl.capabilities import collect_specslo_capabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(collect_specslo_capabilities(args.device, tp_size=args.tp_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
