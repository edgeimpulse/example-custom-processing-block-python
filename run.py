#!/usr/bin/env python3
# CLI scaffolding to run dsp.py's generate_features locally
#
# Usage examples:
#   python3 run.py --features features.txt --frequency 62.5 --axes accX accY accZ --scale-axes 1
#   python3 run.py --features "0.1, 0.2, 0.3, ..." --frequency 62.5 --axes accX accY accZ --scale-axes 1

import sys
import os
import re
import json
import argparse
import inspect
from importlib import metadata

import numpy as np

try:
    from dsp import generate_features
except ImportError as e:
    raise ImportError(
        'dsp.py must define a "generate_features" function, but it could not be imported: ' + str(e)
    ) from e

PARAMETERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.json')
REQUIREMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
REQUIRED_PYTHON_VERSION = (3, 12)

YELLOW = '\033[33m'
RESET = '\033[0m'


def print_warning(message):
    print(YELLOW + 'Warning: ' + message + RESET, file=sys.stderr)


def check_environment():
    if sys.version_info[:2] != REQUIRED_PYTHON_VERSION:
        print_warning(
            'expected Python {}.{}, but running Python {}.{}.{}'.format(
                REQUIRED_PYTHON_VERSION[0], REQUIRED_PYTHON_VERSION[1], *sys.version_info[:3]
            )
        )

    if not os.path.isfile(REQUIREMENTS_FILE):
        return

    with open(REQUIREMENTS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            match = re.match(r'^([A-Za-z0-9_.-]+)==(.+)$', line)
            if not match:
                continue

            package, expected_version = match.group(1), match.group(2)
            try:
                installed_version = metadata.version(package)
            except metadata.PackageNotFoundError:
                print_warning('package "{}" from requirements.txt is not installed'.format(package))
                continue

            if installed_version != expected_version:
                print_warning(
                    'package "{}" version mismatch (requirements.txt wants {}, installed {})'.format(
                        package, expected_version, installed_version
                    )
                )


def param_to_arg_name(param):
    # e.g. "scale-axes" -> "scale_axes"
    return param.replace('-', '_')


def load_parameters():
    with open(PARAMETERS_FILE, 'r') as f:
        parameters = json.loads(f.read())

    items = []
    for group in parameters.get('parameters', []):
        for item in group.get('items', []):
            items.append(item)

    return parameters, items


def validate_parameters_against_function(fn, items):
    sig = inspect.signature(fn)
    fn_params = set(sig.parameters.keys())

    missing = []
    for item in items:
        arg_name = param_to_arg_name(item['param'])
        if arg_name not in fn_params:
            missing.append(item['param'])

    if missing:
        raise ValueError(
            'The following parameters.json params do not map to arguments in generate_features(): '
            + ', '.join(missing)
        )


def cast_param_value(item, raw_value):
    param_type = item.get('type')
    if param_type == 'int':
        return int(raw_value)
    if param_type == 'float':
        return float(raw_value)
    if param_type in ('boolean', 'bool'):
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).lower() in ('1', 'true', 'yes')
    return raw_value


def format_result(result):
    # Keep the JSON human-readable, but collapse the (often large) features array to one line.
    placeholder = '__FEATURES_PLACEHOLDER__'
    features = result['features']
    result_without_features = dict(result)
    result_without_features['features'] = placeholder

    pretty = json.dumps(result_without_features, indent=2)
    return pretty.replace('"' + placeholder + '"', json.dumps(features))


def parse_features(raw):
    # Accept either a path to a file (comma-separated or JSON array), or an inline string of the same.
    if os.path.isfile(raw):
        with open(raw, 'r') as f:
            content = f.read()
    else:
        content = raw

    content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = [float(x) for x in content.split(',') if x.strip() != '']

    return np.array(data, dtype=float)


def build_arg_parser(items):
    parser = argparse.ArgumentParser(description='Run dsp.py generate_features() from the CLI')
    parser.add_argument('--features', required=True,
                         help='Path to a JSON file with a features array, or an inline JSON array')
    parser.add_argument('--frequency', required=True, type=float, help='Sampling frequency')
    parser.add_argument('--axes', required=True, help='Comma-separated axis names, e.g. --axes accX,accY,accZ')

    for item in items:
        parser.add_argument('--' + item['param'], required=True, help=item.get('help', ''))

    return parser


def main():
    check_environment()
    print(file=sys.stderr)

    parameters, items = load_parameters()

    validate_parameters_against_function(generate_features, items)

    parser = build_arg_parser(items)
    args = parser.parse_args()

    raw_data = parse_features(args.features)
    axes = args.axes.split(',')

    call_args = {
        'implementation_version': parameters['version'],
        'draw_graphs': True,
        'raw_data': raw_data,
        'axes': axes,
        'sampling_freq': args.frequency,
    }

    for item in items:
        arg_name = param_to_arg_name(item['param'])
        raw_value = getattr(args, arg_name)
        call_args[arg_name] = cast_param_value(item, raw_value)

    result = generate_features(**call_args)

    if isinstance(result.get('features'), np.ndarray):
        result['features'] = result['features'].tolist()

    print(format_result(result))


if __name__ == '__main__':
    main()
