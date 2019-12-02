from __future__ import print_function, division

from itertools import product

import numpy as np
import pandas as pd


class Value ():
	def __init__(self, name: str, index: int):
		self.name = name
		self.index = index

	def set(self, v_name, v_index):
		self.variable_name = v_name
		self.variable_index = v_index


class ValueSet ():
	def __init__(self):
		super ().__init__ ()
		self.index_dict = {}
		self.name_dict = {}

	def __setitem__(self, key, value: Value):
		assert isinstance (key, tuple)
		assert isinstance (key[0], int)
		assert isinstance (key[1], str)
		index = key[0]
		name = key[1]
		self.index_dict[index] = value
		self.name_dict[name] = value

	def __getitem__(self, item):
		if isinstance (item, int) or isinstance (item, np.int64):
			return self.index_dict[item]
		elif isinstance (item, str):
			return self.index_dict[item]
		elif isinstance (item, Value):
			return item
		elif isinstance (item, tuple):
			assert self.index_dict[item[0]] is self.name_dict[item[1]]
			return self.name_dict[item[1]]
		else:
			raise KeyError

	def __len__(self):
		return self.index_dict.__len__ ()

	def get_index(self, v):
		return self.__getitem__ (v).index

	def get_name(self, v):
		return self.__getitem__ (v).name

	def get_all_names(self):
		return self.name_dict.keys ()

	def get_all_indexes(self):
		return self.index_dict.keys ()

	def get_all(self):
		return self.index_dict.values ()


class Variable ():
	def __init__(self, name: str, index: int, domains: ValueSet):
		self.name = name
		self.index = index
		self.domains = domains
		self.domain_size = domains.__len__ ()
		self.isHidden = False

		self.pos = -1
		self.neg = -1

	def set_label(self, pos: int, neg: int):
		self.pos = pos
		self.neg = neg


class VariableSet ():

	def __init__(self):
		self.index_dict = {}
		self.name_dict = {}
		self.name_index = {}

		self.S_index = -1
		self.Y_index = -1
		# self.cpts = {}
		# self.jpt = pd.DataFrame ()
		self.size = self.index_dict.__len__ ()

	def __setitem__(self, key, value):
		assert isinstance (key, tuple)
		assert isinstance (key[0], int)
		assert isinstance (key[1], str)
		index = key[0]
		name = key[1]
		self.index_dict[index] = value
		self.name_dict[name] = value

		self.name_index[index] = name
		self.name_index[name] = index

	def __getitem__(self, item):
		if isinstance (item, int) or isinstance (item, np.int64):
			return self.index_dict[item]
		elif isinstance (item, str):
			return self.name_dict[item]
		elif isinstance (item, Variable):
			return item
		elif isinstance (item, tuple):
			assert self.index_dict[item[0]] is self.name_dict[item[1]]
			return self.name_dict[item[1]]
		else:
			raise KeyError

	# def get_index(self, v):
	# 	return self.__getitem__ (v).index
	#
	# def get_name(self, v):
	# 	return self.__getitem__ (v).name

	def get_all_indexes(self):
		return list (self.index_dict.keys ())

	def get_all_names(self):
		return list (self.name_dict.keys ())

	def set_sensitive_decision(self, S: int, Y: int):
		self.S_index = S
		self.Y_index = Y

	def __len__(self):
		return self.index_dict.__len__ ()

	def pop(self, key):
		name = ''
		index = 0

		if isinstance (key, int) or isinstance (key, np.int64):
			index = key
			name = self.name_index[index]
		elif isinstance (key, str):
			name = key
			index = self.name_index[name]

		self.index_dict.pop (index)
		self.name_dict.pop (name)


class Event ():
	def __init__(self, d: dict):
		self.dict = {}
		self.name = {}
		for k in d.keys ():
			if isinstance (k, Variable):
				var_index = k.index
				self.name[var_index] = k
			else:
				var_index = k
			if isinstance (d[k], Value):
				val_index = d[k].index
			else:
				val_index = d[k]
			self.dict[var_index] = val_index

	def keys(self):
		return sorted (self.dict.keys ())

	def first_key(self):
		return list (self.dict.keys ())[0]

	def values(self):
		return [self.dict[k] for k in self.keys ()]

	def first_value(self):
		return list (self.dict.values ())[0]

	def __len__(self):
		return self.dict.__len__ ()

	def __add__(self, other):
		return Event ({**self.dict, **other.dict})


class CPTSet ():
	def __init__(self):
		self.index_dict = {}
		self.name_dict = {}

	def __setitem__(self, key, value):
		assert isinstance (key, tuple)
		assert isinstance (key[0], int)
		assert isinstance (key[1], str)
		index = key[0]
		name = key[1]
		self.index_dict[index] = value
		self.name_dict[name] = value

	def __getitem__(self, item):
		if isinstance (item, int):
			return self.index_dict[item]
		elif isinstance (item, str):
			return self.name_dict[item]
		elif isinstance (item, tuple):
			assert self.index_dict[item[0]] is self.name_dict[item[1]]
			return self.name_dict[item[1]]
		else:
			raise KeyError


class CBNModel ():
	def __init__(self, g, v):
		self.index_graph = g
		self.v = v
		self.S_index = -1
		self.Y_index = -1
		self.cpts = CPTSet ()
		self.jpt = pd.DataFrame ()

	def build_cpt(self, v_index):
		# transformation
		var = self.v[v_index]
		var_index = var.index

		parents_index = set (self.index_graph.pred[var_index].keys ())
		values = product (*[self.v[i].domains.get_all_indexes () for i in sorted (parents_index)])
		index = tuple (values)
		cpt = pd.DataFrame (data=np.zeros ((index.__len__ (), var.domain_size)), columns=var.domains.get_all_indexes ())
		if index.__len__ () > 1:
			cpt.index = index
		return cpt

	def find_deterministic_value(self, events: Event, conditions: Event):

		event_index = events.first_key ()
		event_value_index = events.first_value ()

		condition_index = conditions.keys ()
		condition_value_index = list ([self.v[j].domains[k].index for j, k in zip (condition_index, conditions.values ())])

		loc = tuple ([k for k in condition_value_index])
		return np.argmax (self.cpts[event_index].iloc[loc])

	def set_conditional_prob(self, events: Event, conditions: Event, prob: float):
		if events.__len__ () > 1:
			raise KeyError
		# transformation
		event_index = events.first_key ()
		event_value_index = events.first_value ()

		condition_index = conditions.keys ()
		condition_value_index = list ([self.v[j].domains[k].index for j, k in zip (condition_index, conditions.values ())])

		graphical_parents_index = list (self.index_graph.pred[event_index].keys ())
		assert condition_index == graphical_parents_index

		if conditions.__len__ () == 0:
			self.cpts[event_index].loc[0, event_value_index] = prob
		else:
			loc = tuple ([k for k in condition_value_index])
			self.cpts[event_index].loc[loc, event_value_index] = prob

	def build_joint_table(self):
		joint_event = product (*[self.v[i].domains.get_all_indexes () for i in sorted (self.v.get_all_indexes ())])
		self.jpt = pd.DataFrame (data=list (map (list, list (joint_event))), columns=self.v.get_all_indexes ())

		joint_event_prob = pd.DataFrame (data=np.zeros (shape=self.jpt.shape), columns=self.jpt.columns)

		for v_index in self.v.get_all_indexes ():
			parents_index = set (self.index_graph.pred[v_index].keys ())
			joint_event_prob[v_index] = self.jpt.apply (lambda l: self.get_prob (Event ({v_index: l[v_index]}), Event ({key: l[key] for key in parents_index})), axis=1)

		self.jpt['prob'] = joint_event_prob.product (axis=1)

	def _get_marginal_prob(self, event: Event):
		groupby_object = self.jpt.groupby (event.keys ())
		name = tuple (event.values ())
		if name.__len__ () == 1:
			name = name[0]
		return groupby_object.get_group (name)['prob'].sum ()

	def _get_conditional_prob(self, events: Event, conditions: Event):
		# if the conditional probability is identical to the cpt,
		# we can simply retrieve the cpts

		event_index = events.first_key ()
		event_value_index = events.first_value ()
		condition_index = conditions.keys ()
		condition_value_index = list ([self.v[j].domains[k].index for j, k in zip (condition_index, conditions.values ())])
		graphical_parents_index = list (self.index_graph.pred[event_index].keys ())

		if events.__len__ () == 1 and condition_index == graphical_parents_index:
			if conditions.__len__ () == 0:
				return self.cpts[event_index].loc[0, event_value_index]
			else:
				loc = tuple ([k for k in condition_value_index])
				return self.cpts[event_index].loc[loc, event_value_index]
		else:
			n = self._get_marginal_prob (conditions)
			if n == 0.0:
				return 0.0
			else:
				return self._get_marginal_prob (events + conditions) / n

	def get_prob(self, event: Event, conditions=None):
		if conditions is None:
			return self._get_marginal_prob (event)
		else:
			return self._get_conditional_prob (event, conditions)

	def get_loc(self, events):
		event_variable_index = events.keys ()
		event_value_index = list ([self.v[j].domains[k].index for j, k in zip (event_variable_index, events.values ())])
		loc = 0
		for i, j in zip (event_variable_index, event_value_index):
			loc *= self.v[i].domain_size
			loc += j

		return loc
