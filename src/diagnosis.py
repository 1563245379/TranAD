import numpy as np
from sklearn.metrics import ndcg_score
from src.constants import lm

def hit_att(ascore, labels, ps = [100, 150]):
	res = {}
	for p in ps:
		hit_score = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
			if l:
				size = round(p * len(l) / 100)
				a_p = set(a[:size])
				intersect = a_p.intersection(l)
				hit = len(intersect) / len(l)
				hit_score.append(hit)
		res[f'Hit@{p}%'] = np.mean(hit_score)
	return res

def ndcg(ascore, labels, ps = [100, 150]):
	res = {}
	for p in ps:
		ndcg_scores = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			labs = list(np.where(l == 1)[0])
			if labs:
				k_p = round(p * len(labs) / 100)
				try:
					hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
				except Exception as e:
					return {}
				ndcg_scores.append(hit)
		res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
	return res

def hit_att_union(preds, labels, ps=[100, 150]):
	"""Hit@ for union evaluation: check if any predicted anomalous point has any true anomaly."""
	res = {}
	for p in ps:
		hit_scores = []
		# preds: (time_steps, feats), labels: (time_steps, feats)
		for i in range(preds.shape[0]):
			pred_row, label_row = preds[i], labels[i]
			# Anomalous features in ground truth at time i
			true_anomalies = set(np.where(label_row == 1)[0])
			if not true_anomalies:
				continue
			# Anomalous features predicted at time i
			predicted_anomalies = set(np.where(pred_row == 1)[0])
			size = round(p * len(true_anomalies) / 100)
			if size == 0:
				size = 1
			# For union: if we predicted any anomaly, count as hit
			if len(predicted_anomalies) > 0:
				# Check if we catch at least one of the true anomalies
				hits = len(predicted_anomalies.intersection(true_anomalies))
				hit = hits / len(true_anomalies)
			else:
				hit = 0.0
			hit_scores.append(hit)
		res[f'Hit@{p}%'] = np.mean(hit_scores) if hit_scores else 0.0
	return res

def ndcg_union(preds, labels, ps=[100, 150]):
	"""NDCG@ for union evaluation: score based on overlap of predicted and true anomalies."""
	res = {}
	for p in ps:
		ndcg_scores = []
		for i in range(preds.shape[0]):
			pred_row, label_row = preds[i], labels[i]
			true_anomalies = np.where(label_row == 1)[0]
			if len(true_anomalies) == 0:
				continue
			# Build a score vector: higher if feature is both predicted and true anomalous
			score = np.zeros_like(pred_row, dtype=float)
			for feat in range(len(pred_row)):
				if pred_row[feat] == 1 and feat in true_anomalies:
					score[feat] = 1.0
			# Perfect score: all true anomalies predicted
			perfect = np.zeros_like(label_row, dtype=float)
			perfect[true_anomalies] = 1.0
			try:
				ndcg = ndcg_score(perfect.reshape(1, -1), score.reshape(1, -1), k=len(true_anomalies))
			except:
				ndcg = 0.0
			ndcg_scores.append(ndcg)
		res[f'NDCG@{p}%'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
	return res



