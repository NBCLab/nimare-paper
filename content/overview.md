# NiMARE Overview

NiMARE is designed to be modular and object-oriented, with an interface that mimics popular Python libraries, including scikit-learn and nilearn.
This standardized interface allows users to employ a wide range of meta-analytic algorithms without having to familiarize themselves with the idiosyncrasies of algorithm-specific tools.
This lets users use whatever method is most appropriate for a given research question with minimal mental overhead from switching methods.
Additionally, NiMARE emphasizes citability, with references in the documentation and citable boilerplate text that can be copied directly into manuscripts, in order to ensure that the original algorithm developers are appropriately recognized.

NiMARE works with Python versions 3.6 and higher, and can easily be installed with `pip`.
Its source code is housed and version controlled in a GitHub repository at https://github.com/neurostuff/NiMARE.

NiMARE is under continued active development, and we anticipate that the user-facing API (application programming interface) may change over time.
Our emphasis in this paper is thus primarily on reviewing the functionality implemented in the package and illustrating the general interface, and not on providing a detailed and static user guide that will be found within the package documentation.

Tools in NiMARE are organized into several modules, including `nimare.meta`, `nimare.correct`, `nimare.annotate`, `nimare.decode`, and `nimare.workflows`.
In addition to these primary modules, there are several secondary modules for data wrangling and internal helper functions, including `nimare.io`, `nimare.dataset`, `nimare.extract`, `nimare.stats`, `nimare.utils`, and `nimare.base`.
These modules are summarized in [](api.md), as well as in {numref}`table_modules`.
