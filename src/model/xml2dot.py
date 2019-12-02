import re
import xml.etree.ElementTree as XMLParser


def rename_dot_node(name):
	'''
	An ID is one of the following:
	Any string of alphabetic ([a-zA-Z\200-\377]) characters, underscores ('_') or digits ([0-9]), not beginning with a digit;
	a numeral [-]?(.[0-9]+ | [0-9]+(.[0-9]*)? );
	any double-quoted string ("...") possibly containing escaped quotes (\")1;
	an HTML string (<...>).
	'''
	# for simplicity, we remove all non-letter and non-number
	return re.sub ('[^a-z0-9]]', '_', name)


def convert2dot(xml_file, dot_file):
	dot = open (dot_file, 'w')
	dot.write ('strict digraph G{\n')

	bayesian_net = XMLParser.parse (xml_file)
	root = bayesian_net.getroot ()

	# variables
	variables = root[0]
	for v in variables:

		if 'latent' in v.attrib:
			latent = True if v.attrib['latent'] == 'yes' else False
		else:
			latent = False
		if 'name' in v.attrib:
			name = v.attrib['name']
			if latent:
				dot.write ('\t%s [label=<U<SUB>%s</SUB>>, style=dashed];\n' % (name, name[1:]))
			else:
				dot.write ('\t%s [label="%s"];\n' % (name, name))
		else:
			raise KeyError

	# edges and parents
	parents = root[1]
	for variable in parents:
		if 'name' in variable.attrib:
			name = variable.attrib['name']
		else:
			raise KeyError

		for parent in variable:
			if 'name' in parent.attrib:
				parent_name = parent.attrib['name']
			else:
				raise KeyError
			dot.write ('\t%s -> %s;\n' % (parent_name, name))
	dot.write ('}')
	dot.close ()
