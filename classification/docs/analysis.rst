##############################
Classification Result Analysis
##############################

Multiclass metrics in ROC
=========================

Receiver-operating characteric calculations can show the relationship between
the true-positive-rate and the false-positive-rate by varying the
detection-threshold. This is well-defined in a binary-comparison setting, but
can be achieved in multiple ways in a multiclass comparison.

Possibilities for multiclass metrics for ROC-calculation:

- macro
    Calculate for each group separately and then average globally.
- micro
    Sum over all cases individually (eg take tpr and fpr numbers)
