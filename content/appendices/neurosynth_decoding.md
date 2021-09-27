# Appendix II: Neurosynth Discrete Decoding

The implementation of the MKDA Chi-squared meta-analysis method used by Neurosynth is quite similar to BrainMap’s method for decoding, if applied to annotations instead of modeled activation values.
This method compares the distributions of studies with each label within the sample against those in a larger database, but, unlike the BrainMap method, does not take foci into account.
For this reason, the Neurosynth method would likely be more appropriate for selection criteria not based on regions of interest (e.g., for characterizing meta-analytic groupings from a meta-analytic clustering analysis).
However, the Neurosynth method requires user-provided information that BrainMap does not.
Namely, in order to estimate probabilities for the consistency and specificity analyses with Bayes’ Theorem, the Neurosynth method requires a prior probability of a given label.
Typically, a value of 0.5 is used (i.e., the estimated probability that an individual is undergoing a given mental process described by a label, barring any evidence from neuroimaging data, is predicted to be 50%).
This is, admittedly, a poor prediction, which means that probabilities estimated based on this prior are not likely to be accurate, though they may still serve as useful estimates of effect size for the analysis.

Like the BrainMap method, this method produces four outputs for each label.
For the consistency analysis, this method produces both a p-value and a conditional probability of selection given the presence of the label and the prior probability of having the label.
For the specificity analysis, the Neurosynth method produces both a p-value and a posterior probability of presence of the label given selection and the prior probability of having the label.
A detailed algorithm description is presented below.

The Neurosynth method for discrete functional decoding performs both forward and reverse inference using an annotated coordinate-based database and a target sample of studies within that database.
Unlike the BrainMap approach, the Neurosynth approach uses an *a priori* value as the prior probability of any given experiment including a given label.

1.  Select studies in the database according to some criterion (e.g.,
    having at least one peak in an ROI).

2.  For each label, studies in the database can now be divided into four
    groups:
    -   Label-positive and selected --\> $S_{s+l+}$
    -   Label-negative and selected --\> $S_{s+l-}$
    -   Label-positive and unselected --\> $S_{s-l+}$
    -   Label-negative and unselected --\> $S_{s-l-}$

3.  Set a prior probability $p$ of a given mental state occurring in the
    real world.
    -   Neurosynth uses `0.5` as the default.

4.  Compute $P(s^{+})$:
    -   Probability of being selected,
        $P(s^{+}) = S_{s+} / (S_{s+} + S_{s-})$, where
        $S_{s+} = S_{s+l+} + S_{s+l-}$ and
        $S_{s-} = S_{s-l+} + S_{s-l-}$

5.  For each label, compute $P(l^{+})$:
    -   $P(l^{+}) = S_{l+} / (S_{l+} + S_{l-}$, where
        $S_{l+} = S_{s+l+} + S_{s-l+}$ and
        $S_{l-} = S_{s+l-} + S_{s-l-}$

6.  Compute $P(s^{+}|l^{+})$:
    -   $P(s^{+}|l^{+}) = S_{s+l+} / S_{l+}$

7.  Compute $P(s^{+}|l^{-})$:
    -   $P(s^{+}|l^{-}) = S_{s+l-} / S_{l-}$
    -   Only used to determine sign of reverse inference z-value.

8.  Compute $P(s^{+}|l^{+}, p)$, where is the prior probability of a
    label:
    -   This is the forward inference posterior probability. Probability
        of selection given label and given prior probability of label,
        $p$.
    -   $P(s^{+}|l^{+}, p) = pP(s^{+}|l^{+}) + (1 - p)P(s^{+}|l^{-})$

9.  Compute $P(l^{+}|s^{+}, p)$:
    -   This is the reverse inference posterior probability. Probability
        of label given selection and given the prior probability of
        label.
    -   $P(l^{+}|s^{+}, p) = pP(s^{+}|l^{+}) / P(s^{+}|l^{+}, p)$

10. Perform a one-way chi-square test to determine if the rate at which
    studies are selected for a given label is significantly different
    from the average rate at which studies are selected across labels.
    -   Convert p-value to signed z-value using whether the number of
        studies selected for the label is greater than or less than the
        mean number of studies selected across labels to determine the
        sign.

11. Perform a two-way chi-square test to determine if presence of the
    label and selection are independent.
    -   Convert p-value to signed z-value using $P(s^{+}|l^{-})$ to
        determine sign.
