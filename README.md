# Boston housing predictor

THe data is directly fetched from the internet --   ```http://lib.stat.cmu.edu/datasets/boston```. Streamlit was used for ease of use, a proper app would suffer from the lack of customizability it brings.

It is important to note that all the other regressions are just slight improvements to linear regression, random forest is signficant leap forward but xgboost is king.

The aim is to justify linear regressions value as the foundation, but also call tell others that based on its assuptions we would need to be robots for it to work and we are human hence xgboost (for small sample data) and neural nets (for big data) are more realistic.

## How to launch

Start python environment

lsetga@lsetga:~/Projects/pyhon1$  source /home/lsetga/Projects/pyhon1/env/bin/activate

The above is deactivated by typing "deactiavte"

Then

```bash
streamlit run app.py
```

The above is deactivated by pressing ctrl c.

### issues that really bugged me

```bash
TypeError: only 0-dimensional arrays can be converted to Python scalars

File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/lsetga/Projects/pyhon1/app.py", line 766, in <module>
    lin_pred = float(models['lin_model'].predict(input_df))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

```bash
AttributeError: 'numpy.ndarray' object has no attribute 'scatter'

File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/lsetga/Projects/pyhon1/app.py", line 880, in <module>
    axes.scatter(y_train, models['y_train_pred_rf'], alpha=0.6, c=y_train, cmap='viridis')
    ^^^^^^^^^^^^
```

```bash
TypeError: only 0-dimensional arrays can be converted to Python scalars

File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/lsetga/Projects/pyhon1/app.py", line 988, in <module>
    pred_sales = float(pred_sales_raw)
                 ^^^^^^^^^^^^^^^^^^^^^
```

## I would also like some addons

A page that explains the business sense for login and admin. Someone can easily ask the admin to include a specific dataset, specific weight or implement a specific algorithm for that specific user.

I aslo would like the rough mathematical fully worked out example, analogical explanation and senior software engineer breakdown of everything that happened. I expect a more involved, comprehensive and simplistic conclusion with bullet points.

 The aim is to justify linear regressions value as the foundation, but also call tell others that based on its assuptions we would need to be robots for it to work and we are human hence xgboost (for small sample data) and neural nets (for big data) are more realistic.
