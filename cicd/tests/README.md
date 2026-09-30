# Unit tests

These are the original Bojaxns unit/numerical tests, relocated outside the
installed package. Run `conda run -n bojaxns_py python -m pytest cicd/tests`.
System scenarios and explicit performance timing live in their sibling folders.
Generated artifacts belong in pytest's `tmp_path`, not the working directory.
