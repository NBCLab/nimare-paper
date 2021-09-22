# Application Programming Interface

One of the principal goals of NiMARE is to implement a range of methods with a set of shared interfaces, to enable users to employ the most appropriate algorithm for a given question without introducing a steep learning curve.
This approach is modeled on the widely-used scikit-learn package {cite:p}`scikit-learn,sklearn_api`, which implements a large number of machine learning algorithms - all with simple, consistent interfaces.
Regardless of the algorithm employed, data should be in the same format and the same class methods should be called to fit and/or generate predictions from the model.

To this end, we have adopted an object-oriented approach to NiMARE’s core API that organizes tools based on the type of inputs and outputs they operate over.
The key data structure is the `Dataset` class, which stores a range of neuroimaging data amenable to various forms of meta-analysis.
There are two main types of tools that operate on a `Dataset` class.
`Transformer` classes, as their name suggests, perform some transformation on a `Dataset—` i.e., they take a `Dataset` instance as input, and return a modified version of that `Dataset` instance as output (for example, with newly generated maps stored within the object).
`Estimator` classes apply a meta-analytic algorithm to a `Dataset` and return a set of statistical images stored in a MetaResult container class.
The key methods supported by each of these base classes, as well as the main arguments to those methods, are consistent throughout the hierarchy (e.g., all `Transformer` classes must implement a `transform()` method), minimizing the learning curve and ensuring a high degree of predictability for users.
