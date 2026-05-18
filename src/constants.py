import os, json
from src.parser import *
from src.folderconstants import *

# Load params.json overrides
_params_file = os.path.join(os.path.dirname(__file__), 'params.json')
_params_overrides = {}
if os.path.exists(_params_file):
	with open(_params_file) as f:
		_params_overrides = json.load(f)

def _get_file_prefix():
	ds = args.dataset
	if ds == 'energy':
		return (args.file + '_') if args.file else ''
	elif ds == 'SMD': return 'machine-1-1_'
	elif ds == 'SMAP': return (args.file + '_') if args.file else 'P-1_'
	elif ds == 'MSL': return (args.file + '_') if args.file else 'C-1_'
	elif ds == 'UCR': return '136_'
	elif ds == 'NAB': return 'ec2_request_latency_system_failure_'
	return ''

def _get_override(key, default):
	ds = args.dataset
	ds_params = _params_overrides.get(ds, {})
	file_key = _get_file_prefix()
	if file_key in ds_params:
		val = ds_params[file_key].get(key)
		if val is not None:
			return eval(val) if isinstance(val, str) and val.startswith('(') else val
	if '' in ds_params:
		val = ds_params[''].get(key)
		if val is not None:
			return eval(val) if isinstance(val, str) and val.startswith('(') else val
	return default

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
		'MSDS': [(0.91, 1), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
		'energy': [(0.97, 1), (0.99, 1)],
		'PowerSystemAnomalyDetection': [(0.97, 1), (0.99, 1)],
	}
lm = _get_override('lm_d', lm_d[args.dataset][1 if 'TranAD' in args.model else 0])

# Hyperparameters
lr_d = {
		'SMD': 0.0001,
		'synthetic': 0.0001,
		'SWaT': 0.008,
		'SMAP': 0.001,
		'MSL': 0.002,
		'WADI': 0.0001,
		'MSDS': 0.001,
		'UCR': 0.006,
		'NAB': 0.009,
		'MBA': 0.001,
		'energy': 0.001,
		'PowerSystemAnomalyDetection': 0.001,
	}
lr = _get_override('lr_d', lr_d[args.dataset])

# Debugging
percentiles = {
		'SMD': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
		'energy': (97, 500),
		'PowerSystemAnomalyDetection': (97, 500),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9
