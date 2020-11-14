Result Objects
==============

Network File
------------
``network_file_name = "network.tsv"``

The network.tsv is a long-format TSV file containing Regulator -> Target edges.
This TSV file is sorted by the confidence score of the regulator (TF) -> target (gene) edge, from largest to smallest.::

 target	        regulator	combined_confidences	gold_standard	precision	recall	    MCC	        F1
 BSU24750	BSU04730	0.999986                1               1	        0.00165	    0.04057     0.003295
 BSU13020	BSU04730	0.999984
 BSU09690	BSU04730	0.99998
 BSU06590	BSU04730	0.999978
 BSU18510	BSU04730	0.999976
 BSU25800	BSU25810	0.999975

If the gene and TF are in the gold standard, the gold standard for this edge is reported (1 if present, 0 if not present),
and the model performance is calculated. The Precision, Recall, MCC, and F1 scores are calculated assuming that all edges
above a row (with greater confidence scores) are predicted TF -> Gene interactions, and all values below are predicted to
not be TF -> Gene interactions. Rows which do not contain any gold standard (either 1 or 0) indicate that the regulator or
the target are not in the Genes x TFs gold standard matrix. These rows will not be scored.

Also included is a column indicating if the network edge was in the prior (1, 0, or not present if the gene or TF were not
present in the prior network).
The ``beta.sign.sum`` column is the number of times the model coefficient occurred and the sign
(positive model coefficients will be reported as a positive value, and negative model coefficients will be reported as a
negative value).
The ``var.exp.median`` column reports the median amount of variance in the gene explained by the regulator.

InferelatorResults
------------------
.. autoclass:: inferelator.postprocessing.InferelatorResults
   :exclude-members: name, network, betas, betas_sign, betas_stack, combined_confidences, tasks

   .. autoattribute:: name
       :annotation:

   .. autoattribute:: network
       :annotation:

   .. autoattribute:: betas_sign
       :annotation:

   .. autoattribute:: betas_stack
       :annotation:

   .. autoattribute:: combined_confidences
       :annotation:

   .. autoattribute:: tasks
       :annotation:
