# ************ WHAT TYPE OF RESULTS ARE MERGED (name of output file) ************ #
name = "m500b64"

out_filename = "othello-results_{}.json".format(name)
# Create file
open(out_filename, 'a').close()


def cat_json(output_filename, input_filenames):
	"""
	Merges multiple json files into one large json file
	:param output_filename: Name of the output file
	:param input_filenames: Array with names of input files
	:return:
	"""
	with open(output_filename, "w") as outfile:
		first = True
		counter = -1
		for infile_name in input_filenames:
			with open(infile_name) as infile:
				if first:
					outfile.write('{')
					first = False
				else:
					outfile.write(',')
				outfile.write(mangle(infile.read(), counter))
				counter -= 1
		outfile.write('}')


def mangle(s, count):
	# The json files all contain a "-1" key which should be changed when merging
	edit = s.strip()[1:-1]
	edit = edit.replace('"-1"', '"' + str(count) + '"')
	return edit


# The input files all have the format othello_results_XX999.json => XX can be replaced by number
numbers = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]
in_filenames = ['othello_results_{}999.json'.format(number) for number in numbers]
cat_json(out_filename, in_filenames)
