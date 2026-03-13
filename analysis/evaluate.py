"""Evaluate the trained model on a held-out test set.

Runs the RL-trained model on test examples, logs which tool was called for each
question type and game phase, and builds a routing matrix (question_type x tool_name)
to show whether the model learned meaningful tool-selection patterns.
"""
