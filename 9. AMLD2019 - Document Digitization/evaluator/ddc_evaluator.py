import csv

class DdcEvaluator:
  def __init__(self, answer_file_path, round=1):
    """
    `round` : Holds the round for which the evaluation is being done. 
    can be 1, 2...upto the number of rounds the challenge has.
    Different rounds will mostly have different ground truth files.
    """
    self.answer_file_path = answer_file_path
    self.round = round

  def _evaluate(self, client_payload, _context={}):
    """
    `client_payload` will be a dict with (atleast) the following keys :
      - submission_file_path : local file path of the submitted file
      - aicrowd_submission_id : A unique id representing the submission
      - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
    """
    submission_file_path = client_payload["submission_file_path"]
    aicrowd_submission_id = client_payload["aicrowd_submission_id"]
    aicrowd_participant_uid = client_payload["aicrowd_participant_id"]
    
    """
    Do something with your submitted file to come up
    with a score and a secondary score.

    if you want to report back an error to the user,
    then you can simply do :
      `raise Exception("YOUR-CUSTOM-ERROR")`

     You are encouraged to add as many validations as possible
     to provide meaningful feedback to your users
    """
    
    with open(answer_file_path) as answer_file:
        with open(submission_file_path) as submission_file:
            truth = csv.DictReader(answer_file)
            truth_header = truth.fieldnames
            submission = csv.DictReader(submission_file)            
            submission_header = submission.fieldnames
            
            if submission_header != truth_header :
                raise Exception("File format exception : the expected sumission header is:\n" + ",".join(truth_header))
    
            submission_map = {row["companyName"].lower() : row for row in submission}
            incorrect_map = {key : 0 for key in truth_header[1:]}
            
            truth_cells = 0
            correct_cells = 0
            for truth_row in truth:
                submission_row = submission_map.get(truth_row["companyName"].lower())
                if not submission_row:
                    print("Company name " + truth_row["companyName"] + " not found in submission")
                for key in truth_header[1:]:                    
                    truth_cells += 1
                    if not submission_row:
                        continue
                    submission_cell = submission_row[key].lower().strip()
                    truth_cell = truth_row[key].lower().strip()
                    if key in ["canBorrow", "directorsBorrow", "resolutionNeeded"] :                        
                        if submission_cell not in ["", "yes", "no"] :
                            raise Exception("Format exception : expected yes/no for column " + key + ", got " + submission_cell)
                    if submission_cell == truth_cell:
                        correct_cells += 1
                    else :
                        incorrect_map[key] += 1
    
    print("total cells : " + str(truth_cells))
    print("correct cells : " + str(correct_cells))
    print("errors by category : " + str(incorrect_map))
    
    
    _result_object = {
        "score": round(correct_cells/truth_cells*100, 2),
        "score_secondary" : ""
    }
    return _result_object

if __name__ == "__main__":
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    answer_file_path = "data/ground_truth_training.csv"
    _client_payload = {}
    _client_payload["submission_file_path"] = "data/results_training.csv"
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234
    
    # Instaiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = DdcEvaluator(answer_file_path)
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
