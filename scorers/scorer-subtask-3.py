import logging.handlers
import argparse
import sys
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn

"""
Scorer for SemEval 2023 task 3 subtask 3. 
Since it is a multilabel multiclass classification problem, we compute the micro F1 over all the classes. 
One row of the prediction file is the following:
article_id|TAB|row_id|TAB|technique1,technique2,...,techniqueN

where article_id is the numerical code in the name of the file with the input article, |TAB| is the tab character, row_id is the paragraph id (starting from 1) for which predictions are supposed to be made, and technique1,technique2,...,techniqueN is a comma-separated list of techniques (the list could be empty). 
For example:

111111111	1	Loaded_Language
111111111	3	Name_Calling,Whataboutism
111111111	4	Causal_Oversimplification,False_Dilemma-No_Choice,Loaded_Language
111111112	1	
111111112	3	Slogans,Straw_Man
"""

logger = logging.getLogger("task3_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)


def read_techniques_list_from_file(file_full_name):
  """
  Read the list of technique names from a file, one per line. 
  """
  with open(file_full_name, encoding='utf-8') as f:
    return [ line.rstrip() for line in f.readlines() ]

  
def _read_csv_input_file(file_full_name):
  """ 
  Read a csv file with three columns TAB separated:
   - first column is the id of the article
   - second column is the id of the paragraph in the article  
   - third column is the comma-separated list of techniques of the example
  """
  a = {}
  with open(file_full_name, encoding='utf-8') as f:
    for line in f.readlines():
      cols = line.rstrip().split("\t")
      if len(cols) < 2 or len(cols) > 3:
        logger.error('The file is supposed to have three columns TAB separated, %d columns found on line %s'%(len(cols),line))
        sys.exit(1)
      example_id = cols[0]+"_"+cols[1]
      if len(cols)==2: # no techniques
        a[example_id] = []
      else:
        a[example_id] = cols[2].split(",")
  return a


def _labels_correct(labels, CLASSES, debug=False):
  """
  Make sure all the labels correspond to strings in the CLASSES array
  :param labels: a dictionary with strings as values
  :param CLASSES: a list of allowed labels
  """
  if debug:
    s=""
    for articleid in labels.keys():
      s += ",".join([ l for l in labels[articleid] if l not in CLASSES ])
    return s
  else:
    for articleid in labels.keys():
      for l in labels[articleid]:
        if l not in CLASSES:
          return False
  return True


def _correct_number_of_examples(pred_labels, gold_labels, debug=False):
  """
  Make sure that the number of predictions is exactly the same as the gold labels
  """
  if debug:
    return ", ".join(set(pred_labels.keys()).symmetric_difference(set(gold_labels.keys()))).replace("_", " ")
  return len(pred_labels.keys())==len(gold_labels.keys())


def _correct_id_list(pred_labels, gold_labels, debug=False):
  """
  Check that the list of keys of pred_labels is the same as the gold file
  """
  if debug:
    return ", ".join(set(pred_labels.keys()).symmetric_difference(set(gold_labels.keys())))
  return len(set(pred_labels.keys()).symmetric_difference(set(gold_labels.keys())))==0
 

def _extract_matching_lists(pred_labels, gold_labels):
  """
  Extract the list of values from the two dictionaries ensuring that elements with the same key are in the same position.
  """
  pred_values, gold_values = ([],[])
  for k in gold_labels.keys():
    pred_values.append(pred_labels[k])
    gold_values.append(gold_labels[k])
  return pred_values, gold_values


def correct_format(pred_labels, gold_labels, CLASSES):
  """
  Check whether the format of the prediction file is correct. 
  The number of checks that can be performed depends on the availability of the gold labels
  """
  if not _labels_correct(pred_labels, CLASSES):
    logger.error('The following labels in the prediction file are not valid: {}.'
                 .format(_labels_correct(pred_labels, CLASSES, True)))
    return False
  if gold_labels: # we can do further checks if the gold_labels are available
    if not _correct_number_of_examples(pred_labels, gold_labels):
      logger.error('The number of predictions (%d) is not the expected one (%d). Here are the different ids: %s' %(len(pred_labels.keys()), len(gold_labels.keys()), _correct_number_of_examples(pred_labels, gold_labels, True)))
      return False
    if not _correct_id_list(pred_labels, gold_labels):
      logger.error('The list of articles ids is not correct. The following ids are not in the gold file: %s'
                   %(_correct_id_list(pred_labels, gold_labels, True)))
      return False
  return True


def evaluate(pred_labels, gold_labels, CLASSES):
  """
    Evaluates the predicted classes w.r.t. a gold file.
    Metrics are: multilabel macro_f1 nd micro_f1
    :param pred_labels: a dictionary with predictions, 
    :param gold_labels: a dictionary with gold labels.
  """
  pred_values, gold_values = _extract_matching_lists(pred_labels, gold_labels)  
  mlb = MultiLabelBinarizer()
  mlb.fit([CLASSES])
  gold_values = mlb.transform(gold_values)
  pred_values = mlb.transform(pred_values)

  macro_f1 = f1_score(gold_values, pred_values, average="macro", zero_division=1)
  micro_f1 = f1_score(gold_values, pred_values, average="micro", zero_division=1)
  return macro_f1, micro_f1


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--gold_file_path", '-g', type=str, required=False, help="Paths to the file with gold annotations.")
  parser.add_argument("--pred_file_path", '-p', type=str, required=True, help="Path to the file with predictions")
  parser.add_argument("--techniques_file_path", '-f', type=str, required=True, help="Path to the file with the names of the techniques")
  parser.add_argument("--log_to_file", "-l", action='store_true', default=False,
                      help="Set flag if you want to log the execution file. The log will be appended to <pred_file>.log")
  parser.add_argument('--output-for-script', "-o", dest='output_for_script', required=False, action='store_true',
                      default=False, help="Prints the output in a format easy to parse for a script")
  args = parser.parse_args()

  output_for_script = bool(args.output_for_script)
  if not output_for_script:
    logger.addHandler(ch)
  
  CLASSES = read_techniques_list_from_file(args.techniques_file_path)

  pred_file = args.pred_file_path
  if args.gold_file_path:
    gold_file = args.gold_file_path
  else:
    gold_file = None
  if args.log_to_file:
    output_log_file = pred_file + ".log"
    logger.info("Logging execution to file " + output_log_file)
    fileLogger = logging.FileHandler(output_log_file)
    fileLogger.setLevel(logging.DEBUG)
    fileLogger.setFormatter(formatter)
    logger.addHandler(fileLogger)
    logger.setLevel(logging.DEBUG) #

  if args.log_to_file:
    logger.info('Reading predictions file') 
  else:
    logger.info('Reading predictions file {}'.format(args.pred_file_path))
  if gold_file:
    if args.log_to_file:
      logger.info('Reading gold file')
    else:
      logger.info("Reading gold predictions from file {}".format(args.gold_file_path))
  else:
    logger.info('No gold file provided')

  pred_labels = _read_csv_input_file(pred_file)    
  gold_labels = _read_csv_input_file(gold_file) if gold_file else None
    
  if correct_format(pred_labels, gold_labels, CLASSES):
    logger.info('Prediction file format is correct')
    if gold_labels:
      macro_f1, micro_f1 = evaluate(pred_labels, gold_labels, CLASSES)
      logger.info("micro-F1={:.5f}\tmacro-F1={:.5f}".format(micro_f1, macro_f1))
      results_by_class = []
      for c in CLASSES:
          y_true_c = [(1 if c in v else 0) for k, v in gold_labels.items()]
          y_pred_c = [(1 if c in v else 0) for k, v in pred_labels.items()]
          c_fscore = f1_score(y_true_c, y_pred_c, average='binary')
          c_precision = precision_score(y_true_c, y_pred_c, average='binary')
          c_recall = recall_score(y_true_c, y_pred_c, average='binary')
          results_by_class.append((c, c_fscore, c_precision, c_recall, sum(y_true_c)))
          
      results_by_class.sort(key=lambda tup: tup[-1]) 
      for result_by_class in results_by_class:
          print(result_by_class)
      if output_for_script:
        print("{}\t{}".format(micro_f1, macro_f1))

