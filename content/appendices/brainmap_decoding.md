# Appendix I: BrainMap Discrete Decoding

The BrainMap discrete decoding method compares the distributions of studies with each label within the sample against those in a larger database while accounting for the number of foci from each study.
Broadly speaking, this method assumes that the selection criterion is associated with one peak per study, which means that it is likely only appropriate for selection criteria based around foci, such as regions of interest.
One common analysis, meta-analytic clustering, involves dividing studies within a database into meta-analytic groupings based on the spatial similarity of their modeled activation maps (i.e., study-wise pseudo-statistical maps produced by convolving coordinates with a kernel).
The resulting sets of studies are often functionally decoded in order to build a functional profile associated with each meta-analytic grouping.
While these groupings are defined as subsets of the database, they are not selected based on the location of an individual peak, and so weighting based on the number of foci would be inappropriate.

This decoding method produces four outputs for each label.
First, the distribution of studies in the sample with the label are compared to the distributions of other labels within the sample.
This consistency analysis produces both a measure of statistical significance (i.e., a p-value) and a measure of effect size (i.e., the likelihood of being selected given the presence of the label).
Next, the studies in the sample are compared to the studies in the rest of the database.
This specificity analysis produces a p-value and an effect size measure of the posterior probability of having the label given selection into the sample.
A detailed algorithm description is presented below.

The BrainMap method for discrete functional decoding performs both forward and reverse inference using an annotated coordinate-based database and a target sample of studies within that database.
Unlike the Neurosynth approach, the BrainMap approach incorporates information about the number of foci associated with each study in the database.

1.  Select studies in the database according to some criterion (e.g.,
    having at least one peak in an ROI).

2.  For each label, studies in the database can now be divided into four
    groups.
    -   Label-positive and selected --\> $S_{s+l+}$
    -   Label-negative and selected --\> $S_{s+l-}$
    -   Label-positive and unselected --\> $S_{s-l+}$
    -   Label-negative and unselected --\> $S_{s-l-}$

3.  Additionally, the number of foci associated with each of these
    groups is extracted.
    -   Number of foci from studies with label, $F_{l+}$
    -   Number of foci from studies without label, $F_{l-}$
    -   Total number of foci in the database, $F_{db} = F_{l+} + F_{l-}$

4.  Compute the number of times any label is used in the database,
    $L_{db}$ (e.g., if every experiment in the database uses two labels,
    then this number is $2S_{db}$, where $S_{db}$ is the total number of
    experiments in the database).

5.  Compute the probability of being selected, $P(s^{+})$.
    -   $P(s^{+}) = S_{s+} / F_{db}$, where
        $S_{s+} = S_{s+l+} + S_{s+l-}$

6.  For each label, compute the probability of having the label,
    $P(l^{+})$.
    -   $P(l^{+}) = S_{l+} / L_{db}$, where
        $S_{l+} = S_{s+l+} + S_{s-l+}$

7.  For each label, compute the probability of being selected given
    presence of the label, $P(s^{+}|l^{+})$.
    -   Can be re-interpreted as the probability of activating the ROI
        given a mental state.
    -   $P(s^{+}|l^{+}) = S_{s+l+} / F_{l+}$

8.  Convert $P(s^{+}|l^{+})$ into the forward inference likelihood,
    $\mathcal{L}$.
    -   $\mathcal{L} = P(s^{+}|l^{+}) / P(s^{+})$

9.  Compute the probability of the label given selection,
    $P(l^{+}|s^{+})$.
    -   Can be re-interpreted as probability of a mental state given
        activation of the ROI.
    -   $P(l^{+}|s^{+}) = \frac{P(s^{+}|l^{+})P(l^{+})}{P(s^{+})}$
    -   This is the reverse inference posterior probability.

10. Perform a binomial test to determine if the rate at which studies
    are selected from the set of studies with the label is significantly
    different from the base probability of studies being selected across
    the whole database.
    -   The number of successes is $\mathcal{K} = S_{s+l+}$, the number
        of trials is
        $n = F_{l+}$, and the hypothesized probability of success is
        $p = P(s^{+})$
    -   If $S_{s+l+} < 5$, override the p-value from this test with 1,
        essentially ignoring
        this label in the analysis.
    -   Convert p-value to unsigned z-value.

11. Perform a two-way chi-square test to determine if presence of the
    label and selection are independent.
    -   If $S_{s+l+} < 5$, override the p-value from this test with 1,
        essentially ignoring this label in the analysis.
    -   Convert p-value to unsigned z-value.
