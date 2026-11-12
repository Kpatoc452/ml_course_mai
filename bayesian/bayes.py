"""Небольшая дискретная байесовская сеть: CPT и точный вывод перебором."""

from graphlib import TopologicalSorter
from itertools import product

import numpy as np
import pandas as pd


class DiscreteBayesNet:
    """Dirichlet(alpha, ..., alpha) для каждой конфигурации родителей.

    Полный перебор подходит только для маленьких учебных сетей.
    Порядок ключей parents не обязан быть топологическим.
    """

    def __init__(self, states, parents, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha должен быть положительным конечным числом")
        if set(states) != set(parents):
            raise ValueError("Для каждой переменной нужны состояния и родители")
        if any(not s or len(s) != len(set(s)) for s in states.values()):
            raise ValueError("Состояния должны быть непустыми и уникальными")
        for node, ps in parents.items():
            if len(ps) != len(set(ps)) or not set(ps) <= set(states):
                raise ValueError(f"Некорректные родители {node}")
        self.order = list(TopologicalSorter(parents).static_order())
        self.states = {v: list(s) for v, s in states.items()}
        self.parents = {v: list(ps) for v, ps in parents.items()}
        self.alpha = float(alpha)

    def fit(self, data):
        if len(data) == 0:
            raise ValueError("Пустые данные")
        self.cpts = {}
        lookup = {}
        for node in self.order:
            if data[node].isna().any() or not set(data[node]) <= set(self.states[node]):
                raise ValueError(f"Пропуски или неизвестные состояния {node}")
            ps = self.parents[node]
            counts = data.groupby(ps + [node], observed=True).size().to_dict()
            rows, lookup[node] = [], {}
            for values in product(*(self.states[p] for p in ps)):
                def count(state):
                    key = values + (state,)
                    return int(counts.get(key if ps else state, 0))

                total = sum(count(s) for s in self.states[node])
                for state in self.states[node]:
                    n = count(state)
                    probability = (n + self.alpha) / (total + self.alpha * len(self.states[node]))
                    rows.append(dict(zip(ps, values)) | {
                        node: state, "n": n, "N_parent": total,
                        "mle": n / total if total else np.nan,
                        "probability": probability,
                    })
                    lookup[node][values + (state,)] = probability
            self.cpts[node] = pd.DataFrame(rows)

        # P(x1, ..., xk) = произведение P(xi | parents(xi)).
        assignments = pd.DataFrame(product(*(self.states[v] for v in self.order)), columns=self.order)
        probabilities = np.ones(len(assignments))
        for node in self.order:
            keys = assignments[self.parents[node] + [node]].itertuples(index=False, name=None)
            probabilities *= np.array([lookup[node][key] for key in keys])
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
            raise ArithmeticError("Совместное распределение не нормировано")
        self.joint = assignments.assign(probability=probabilities)
        return self

    def query(self, target, evidence=None):
        """P(target | evidence), суммирование по всем скрытым переменным."""
        if target not in self.states:
            raise ValueError(f"Неизвестная переменная {target}")
        evidence = {} if evidence is None else evidence
        subset = self.joint
        for node, state in evidence.items():
            if node not in self.states or state not in self.states[node]:
                raise ValueError(f"Неизвестное evidence: {node}={state}")
            subset = subset.loc[subset[node] == state]
        result = subset.groupby(target, observed=True).probability.sum().reindex(self.states[target], fill_value=0)
        return result / result.sum()

    def evidence_probability(self, evidence):
        subset = self.joint
        for node, state in evidence.items():
            if node not in self.states or state not in self.states[node]:
                raise ValueError(f"Неизвестное evidence: {node}={state}")
            subset = subset.loc[subset[node] == state]
        return float(subset.probability.sum())
