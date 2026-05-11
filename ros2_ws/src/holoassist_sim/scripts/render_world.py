#!/usr/bin/env python3
"""Render Jinja2 templates (SDF world, bridge config) from sim_params.yaml.

Usage:
    render_world.py PARAMS_YAML TEMPLATE_FILE OUTPUT_FILE
"""
import os
import sys

import jinja2
import yaml


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2

    params_path, template_path, output_path = sys.argv[1:4]

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    template_dir, template_name = os.path.split(os.path.abspath(template_path))
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    rendered = env.get_template(template_name).render(**params)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(rendered)

    print(f"[render_world] {template_name} -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
