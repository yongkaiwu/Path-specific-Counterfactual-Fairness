import xml.etree.ElementTree as XMLParser

import networkx as nx

from model.variable import Value, ValueSet, Variable, VariableSet, CBNModel


def load_xml_to_cbn(xml_name):
	bayesian_net = XMLParser.parse (xml_name)
	root = bayesian_net.getroot ()

	# vars
	variable_set = VariableSet ()

	for v in root[0]:
		v_name = v.attrib['name']
		v_index = int (v.attrib['index'])

		domains = ValueSet ()
		# values of each vars
		for c in v:
			c_name = c.attrib['name']
			c_index = int (c.attrib['index'])
			domains[c_index, c_name] = Value (c_name, c_index)

		variable_set[(v_index, v_name)] = Variable (v_name, v_index, domains)

	# causal graph
	g = nx.DiGraph ()

	# parents
	for i, pf in enumerate (root[1]):
		pf_name = pf.attrib['name']
		pf_index = variable_set[pf_name].index
		# assert variable_set[pf_index].name == pf_name
		for p in pf:
			p_name = p.attrib['name']
			# int (p.attrib['index']) is the local index
			p_index = variable_set[p_name].index
			g.add_edge (p_index, pf_index)

	cbn = CBNModel (g, variable_set)

	# cpts
	for c in root[2]:
		name = c.attrib['variable']
		v = cbn.v[name]
		index = v.index

		num_row = int (c.attrib['numRows'])
		num_col = int (c.attrib['numCols'])

		cpt = cbn.build_cpt (v)

		assert num_col == v.domain_size
		assert num_row == cpt.shape[0]

		for i, r in enumerate (c):
			cpt.iloc[i, :] = r.text.split (' ')
		cbn.cpts[(index, name)] = cpt.astype (float)

	return cbn
