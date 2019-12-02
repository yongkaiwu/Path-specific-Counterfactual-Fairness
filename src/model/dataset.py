from __future__ import print_function, division

import random

import pandas as pd

from model.variable import Event


def convert(event: Event):
	for k in event.keys ():
		if isinstance (k, int):
			name = event.name[k].name
			v = event.dict[k]
			event.dict.pop (k)
			event.dict[name] = v


class DataSet ():
	def __init__(self, df):
		self.df = df[sorted (df.columns)]
		random.seed (2018)

	def get_marginal_prob(self, event: Event):
		convert (event)
		groupby_object = self.df.groupby (event.keys ())
		name = tuple (event.values ())
		if name.__len__ () == 1:
			name = name[0]
		try:
			return groupby_object.get_group (name).__len__ ()
		except:
			return 0

	def get_conditional_prob(self, events: Event, conditions: Event):
		convert (events)
		convert (conditions)
		n = self.get_marginal_prob (conditions)
		if n == 0.0:
			return 0.0
		else:
			return self.get_marginal_prob (events + conditions) / n

	def count(self, event: Event, condition: Event, relationship: str):
		convert (event)
		convert (condition)
		groupby_object = self.df.groupby (condition.keys ())
		name = tuple (condition.values ())
		if name.__len__ () == 1:
			name = name[0]
		try:
			group = groupby_object.get_group (name)
			key0 = event.keys ()[0]
			key1 = event.keys ()[1]
			if relationship == 'equal':
				sub_group = group[group[key0] == group[key1]]
				return sub_group.__len__ () / group.__len__ ()
			else:
				sub_group = group[group[key0] != group[key1]]
				return sub_group.__len__ () / group.__len__ ()
		except:
			return 0

	def random_assign(self, event: Event, target: Event, p: float):
		convert (event)
		convert (target)
		groupby_object = self.df.groupby (event.keys ())
		name = tuple (event.values ())
		if name.__len__ () == 1:
			name = name[0]
		try:
			index = groupby_object.get_group (name).index
			target_index = random.sample (list (index), int (round (index.__len__ () * p)))
			col = self.df.columns.get_loc (target.keys ()[0])
			self.df.ix[target_index, col] = target.values ()[0]
		except KeyError:
			pass


if __name__ == '__main__':
	data = DataSet (pd.read_csv ('../../data/synthetic/DeterministicData.txt', sep='\t'))
	print (data.get_conditional_prob (Event ({'Y': 1}), Event ({'A': 0, 'N': 0, 'M': 0, 'S': 0})))
	pass
