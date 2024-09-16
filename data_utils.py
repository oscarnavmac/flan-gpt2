

def format_example(example, patterns_list, i):
  inputs_pattern, targets_pattern = patterns_list[i]
  format_strings = {'prompt': inputs_pattern, 'completion': targets_pattern}
  new_example = {}
  for f_name, format_str in format_strings.items():
    new_example[f_name] = format_str.format(**example)
  return new_example